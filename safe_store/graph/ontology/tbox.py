from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from typing import Literal as TypeLiteral
import pipmaster as pm
from ascii_colors import ASCIIColors
from ...core.exceptions import ConfigurationError

try:
    import rdflib
    from rdflib import Graph, URIRef, RDF, RDFS, OWL
    from rdflib import Literal as RDFLiteral
except ImportError:
    pm.ensure_packages(["rdflib"])
    import rdflib
    from rdflib import Graph, URIRef, RDF, RDFS, OWL
    from rdflib import Literal as RDFLiteral

CORE_ONTOLOGY_TURTLE = """
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix ont: <http://example.org/ontology/> .
@prefix ex: <http://example.org/> .

# Core Classes
ont:Agent a owl:Class ;
    rdfs:label "Agent" ;
    rdfs:comment "An autonomous entity, individual, or organization." .

ont:Person a owl:Class ;
    rdfs:subClassOf ont:Agent ;
    rdfs:label "Person" ;
    owl:disjointWith ont:Organization .

ont:Organization a owl:Class ;
    rdfs:subClassOf ont:Agent ;
    rdfs:label "Organization" ;
    owl:disjointWith ont:Person .

ont:Company a owl:Class ;
    rdfs:subClassOf ont:Organization ;
    rdfs:label "Company" .

ont:Document a owl:Class ;
    rdfs:label "Document" ;
    owl:disjointWith ont:Person .

ont:Concept a owl:Class ;
    rdfs:label "Concept" .

ont:Tool a owl:Class ;
    rdfs:label "Tool" .

ont:Location a owl:Class ;
    rdfs:label "Location" .

ont:Event a owl:Class ;
    rdfs:label "Event" .

# Core Relational Properties
ont:worksFor a owl:ObjectProperty ;
    rdfs:domain ont:Person ;
    rdfs:range ont:Organization ;
    owl:inverseOf ont:employs .

ont:employs a owl:ObjectProperty ;
    rdfs:domain ont:Organization ;
    rdfs:range ont:Person ;
    owl:inverseOf ont:worksFor .

ont:collaboratesWith a owl:ObjectProperty, owl:SymmetricProperty ;
    rdfs:domain ont:Agent ;
    rdfs:range ont:Agent .

ont:partOf a owl:ObjectProperty, owl:TransitiveProperty ;
    owl:inverseOf ont:hasPart .

ont:hasPart a owl:ObjectProperty, owl:TransitiveProperty ;
    owl:inverseOf ont:partOf .

ont:locatedIn a owl:ObjectProperty, owl:TransitiveProperty ;
    rdfs:range ont:Location .

ont:dependsOn a owl:ObjectProperty, owl:TransitiveProperty .

ont:hasPrimaryEmail a owl:DatatypeProperty, owl:FunctionalProperty ;
    rdfs:domain ont:Agent ;
    rdfs:range xsd:string .
"""


class TBoxManager:
    """
    Manages the Terminological Component (TBox) / Ontology schema of a knowledge base.
    Provides class hierarchy resolution, domain/range constraints, prompt schema serialization,
    and built-in core ontological axioms (disjointness, symmetry, transitivity, inverses).
    """

    def __init__(self, base_uri: str = "http://example.org/ontology/", load_core: bool = True):
        self.base_uri = base_uri
        self.graph = Graph()
        self._classes: Set[str] = set()
        self._object_properties: Set[str] = set()
        self._datatype_properties: Set[str] = set()

        if load_core:
            self.load_core_ontology()

    def load_core_ontology(self) -> None:
        """Loads SafeStore's built-in standard semantic ontology axioms."""
        try:
            self.graph.parse(data=CORE_ONTOLOGY_TURTLE, format="turtle")
            self._rebuild_indices()
            ASCIIColors.info(f"Loaded core ontology vocabulary ({len(self._classes)} classes, {len(self._object_properties)} relations).")
        except Exception as e:
            ASCIIColors.warning(f"Could not load core ontology: {e}")

    def declare_disjoint(self, class_a: Union[str, URIRef], class_b: Union[str, URIRef]) -> None:
        """Declares two classes as strictly mutually exclusive (owl:disjointWith)."""
        ref_a = URIRef(class_a) if isinstance(class_a, str) else class_a
        ref_b = URIRef(class_b) if isinstance(class_b, str) else class_b
        self.graph.add((ref_a, OWL.disjointWith, ref_b))
        self.graph.add((ref_b, OWL.disjointWith, ref_a))

    def declare_inverse(self, prop_a: Union[str, URIRef], prop_b: Union[str, URIRef]) -> None:
        """Declares two properties as mutual inverses (owl:inverseOf)."""
        ref_a = URIRef(prop_a) if isinstance(prop_a, str) else prop_a
        ref_b = URIRef(prop_b) if isinstance(prop_b, str) else prop_b
        self.graph.add((ref_a, OWL.inverseOf, ref_b))
        self.graph.add((ref_b, OWL.inverseOf, ref_a))

    def declare_symmetric(self, prop: Union[str, URIRef]) -> None:
        """Declares a property as symmetric (owl:SymmetricProperty)."""
        ref = URIRef(prop) if isinstance(prop, str) else prop
        self.graph.add((ref, RDF.type, OWL.SymmetricProperty))

    def declare_transitive(self, prop: Union[str, URIRef]) -> None:
        """Declares a property as transitive (owl:TransitiveProperty)."""
        ref = URIRef(prop) if isinstance(prop, str) else prop
        self.graph.add((ref, RDF.type, OWL.TransitiveProperty))

    def declare_functional(self, prop: Union[str, URIRef]) -> None:
        """Declares a property as functional (at most one value per subject)."""
        ref = URIRef(prop) if isinstance(prop, str) else prop
        self.graph.add((ref, RDF.type, OWL.FunctionalProperty))

    def declare_equivalent(
        self,
        entity_a: Union[str, URIRef],
        entity_b: Union[str, URIRef],
        kind: TypeLiteral['class', 'property', 'individual'] = 'class'
    ) -> None:
        """Declares equivalence between classes, properties, or individuals."""
        ref_a = URIRef(entity_a) if isinstance(entity_a, str) else entity_a
        ref_b = URIRef(entity_b) if isinstance(entity_b, str) else entity_b

        if kind == 'class':
            self.graph.add((ref_a, OWL.equivalentClass, ref_b))
        elif kind == 'property':
            self.graph.add((ref_a, OWL.equivalentProperty, ref_b))
        else:
            self.graph.add((ref_a, OWL.sameAs, ref_b))

    def add_class(
        self,
        class_name_or_uri: str,
        label: Optional[str] = None,
        comment: Optional[str] = None,
        sub_class_of: Optional[str] = None,
        disjoint_with: Optional[str] = None
    ) -> URIRef:
        """Adds a standardized OWL Class to the TBox schema."""
        uri_str = class_name_or_uri if class_name_or_uri.startswith("http") else f"{self.base_uri}{class_name_or_uri}"
        class_ref = URIRef(uri_str)
        self.graph.add((class_ref, RDF.type, OWL.Class))

        lbl = label or class_name_or_uri.split('/')[-1].split('#')[-1]
        self.graph.add((class_ref, RDFS.label, RDFLiteral(lbl)))

        if comment:
            self.graph.add((class_ref, RDFS.comment, RDFLiteral(comment)))

        if sub_class_of:
            parent_ref = URIRef(sub_class_of if sub_class_of.startswith("http") else f"{self.base_uri}{sub_class_of}")
            self.graph.add((class_ref, RDFS.subClassOf, parent_ref))

        if disjoint_with:
            disjoint_ref = URIRef(disjoint_with if disjoint_with.startswith("http") else f"{self.base_uri}{disjoint_with}")
            self.declare_disjoint(class_ref, disjoint_ref)

        self._rebuild_indices()
        return class_ref

    def add_object_property(
        self,
        prop_name_or_uri: str,
        domain: Optional[str] = None,
        range_: Optional[str] = None,
        label: Optional[str] = None,
        is_transitive: bool = False,
        is_symmetric: bool = False,
        is_functional: bool = False,
        inverse_of: Optional[str] = None
    ) -> URIRef:
        """Adds a standardized OWL Object Property connecting classes."""
        uri_str = prop_name_or_uri if prop_name_or_uri.startswith("http") else f"{self.base_uri}{prop_name_or_uri}"
        prop_ref = URIRef(uri_str)
        self.graph.add((prop_ref, RDF.type, OWL.ObjectProperty))

        lbl = label or prop_name_or_uri.split('/')[-1].split('#')[-1]
        self.graph.add((prop_ref, RDFS.label, RDFLiteral(lbl)))

        if domain:
            dom_ref = URIRef(domain if domain.startswith("http") else f"{self.base_uri}{domain}")
            self.graph.add((prop_ref, RDFS.domain, dom_ref))

        if range_:
            rng_ref = URIRef(range_ if range_.startswith("http") else f"{self.base_uri}{range_}")
            self.graph.add((prop_ref, RDFS.range, rng_ref))

        if is_transitive:
            self.declare_transitive(prop_ref)
        if is_symmetric:
            self.declare_symmetric(prop_ref)
        if is_functional:
            self.declare_functional(prop_ref)

        if inverse_of:
            inv_ref = URIRef(inverse_of if inverse_of.startswith("http") else f"{self.base_uri}{inverse_of}")
            self.declare_inverse(prop_ref, inv_ref)

        self._rebuild_indices()
        return prop_ref

    def add_datatype_property(
        self,
        prop_name_or_uri: str,
        domain: Optional[str] = None,
        range_data_type: str = "xsd:string",
        label: Optional[str] = None,
        is_functional: bool = False
    ) -> URIRef:
        """Adds a Datatype Property (attribute) defined on a class."""
        uri_str = prop_name_or_uri if prop_name_or_uri.startswith("http") else f"{self.base_uri}{prop_name_or_uri}"
        prop_ref = URIRef(uri_str)
        self.graph.add((prop_ref, RDF.type, OWL.DatatypeProperty))

        lbl = label or prop_name_or_uri.split('/')[-1].split('#')[-1]
        self.graph.add((prop_ref, RDFS.label, RDFLiteral(lbl)))

        if domain:
            dom_ref = URIRef(domain if domain.startswith("http") else f"{self.base_uri}{domain}")
            self.graph.add((prop_ref, RDFS.domain, dom_ref))

        xsd_map = {
            "xsd:string": URIRef("http://www.w3.org/2001/XMLSchema#string"),
            "xsd:integer": URIRef("http://www.w3.org/2001/XMLSchema#integer"),
            "xsd:float": URIRef("http://www.w3.org/2001/XMLSchema#float"),
            "xsd:boolean": URIRef("http://www.w3.org/2001/XMLSchema#boolean"),
            "xsd:dateTime": URIRef("http://www.w3.org/2001/XMLSchema#dateTime"),
        }
        rng_ref = xsd_map.get(range_data_type, URIRef("http://www.w3.org/2001/XMLSchema#string"))
        self.graph.add((prop_ref, RDFS.range, rng_ref))

        if is_functional:
            self.declare_functional(prop_ref)

        self._rebuild_indices()
        return prop_ref

    def delete_element(self, element_uri: str) -> int:
        """Removes an element and all associated triples from the ontology."""
        ref = URIRef(element_uri)
        removed_count = 0
        for s, p, o in list(self.graph.triples((ref, None, None))):
            self.graph.remove((s, p, o))
            removed_count += 1
        for s, p, o in list(self.graph.triples((None, None, ref))):
            self.graph.remove((s, p, o))
            removed_count += 1
        self._rebuild_indices()
        return removed_count

    def serialize_ontology(self, format: str = "turtle") -> str:
        """Serializes the complete TBox ontology into a Turtle string."""
        return self.graph.serialize(format=format)

    def load_ontology(self, source: Union[str, Path], format: str = "turtle") -> None:
        """Loads an ontology from a file or string."""
        try:
            if isinstance(source, Path) or (isinstance(source, str) and Path(source).is_file()):
                self.graph.parse(location=str(source), format=format)
            else:
                self.graph.parse(data=str(source), format=format)
            self._rebuild_indices()
            ASCIIColors.info(f"Loaded ontology into TBox: {len(self.graph)} triples, {len(self._classes)} classes.")
        except Exception as e:
            raise ConfigurationError(f"Failed to load TBox ontology: {e}") from e

    def _rebuild_indices(self) -> None:
        self._classes.clear()
        self._object_properties.clear()
        self._datatype_properties.clear()

        for s in self.graph.subjects(RDF.type, OWL.Class):
            self._classes.add(str(s))
        for s in self.graph.subjects(RDF.type, RDFS.Class):
            self._classes.add(str(s))

        for s in self.graph.subjects(RDF.type, OWL.ObjectProperty):
            self._object_properties.add(str(s))
        for s in self.graph.subjects(RDF.type, OWL.DatatypeProperty):
            self._datatype_properties.add(str(s))
        for s in self.graph.subjects(RDF.type, RDF.Property):
            if str(s) not in self._object_properties:
                self._datatype_properties.add(str(s))

    def get_classes(self) -> List[str]:
        """Returns all class URIs defined in the TBox."""
        return sorted(list(self._classes))

    def get_subclasses(self, class_uri: str, recursive: bool = True) -> List[str]:
        """Returns subclasses of the given class URI."""
        target_ref = URIRef(class_uri)
        subclasses = set()

        def _traverse(cur_ref):
            for sub in self.graph.subjects(RDFS.subClassOf, cur_ref):
                sub_str = str(sub)
                if sub_str not in subclasses:
                    subclasses.add(sub_str)
                    if recursive:
                        _traverse(sub)

        _traverse(target_ref)
        return sorted(list(subclasses))

    def get_superclasses(self, class_uri: str, recursive: bool = True) -> List[str]:
        """Returns superclasses of the given class URI."""
        target_ref = URIRef(class_uri)
        superclasses = set()

        def _traverse(cur_ref):
            for sup in self.graph.objects(cur_ref, RDFS.subClassOf):
                sup_str = str(sup)
                if sup_str not in superclasses:
                    superclasses.add(sup_str)
                    if recursive:
                        _traverse(sup)

        _traverse(target_ref)
        return sorted(list(superclasses))

    def get_property_domain_range(self, property_uri: str) -> Tuple[Optional[str], Optional[str]]:
        """Returns the (domain, range) URIs for a given property URI."""
        prop_ref = URIRef(property_uri)
        domain = None
        range_ = None

        domain_obj = self.graph.value(subject=prop_ref, predicate=RDFS.domain)
        if domain_obj:
            domain = str(domain_obj)

        range_obj = self.graph.value(subject=prop_ref, predicate=RDFS.range)
        if range_obj:
            range_ = str(range_obj)

        return domain, range_

    def to_prompt_schema(self) -> str:
        """Serializes the TBox classes and properties for LLM graph extraction guidance."""
        lines = ["ONTOLOGY TBOX SCHEMA:"]
        
        lines.append("CLASSES:")
        for cls in self.get_classes():
            label = cls.split("/")[-1].split("#")[-1]
            subclasses = [s.split("/")[-1].split("#")[-1] for s in self.get_subclasses(cls, recursive=False)]
            sub_str = f" (Subclasses: {', '.join(subclasses)})" if subclasses else ""
            lines.append(f"  - {label} ({cls}){sub_str}")

        lines.append("\nRELATIONSHIPS (Object Properties):")
        for prop in sorted(list(self._object_properties)):
            p_label = prop.split("/")[-1].split("#")[-1]
            dom, rng = self.get_property_domain_range(prop)
            dom_l = dom.split("/")[-1].split("#")[-1] if dom else "Any"
            rng_l = rng.split("/")[-1].split("#")[-1] if rng else "Any"
            lines.append(f"  - {p_label}: {dom_l} -> {rng_l} ({prop})")

        lines.append("\nATTRIBUTES (Datatype Properties):")
        for prop in sorted(list(self._datatype_properties)):
            p_label = prop.split("/")[-1].split("#")[-1]
            dom, rng = self.get_property_domain_range(prop)
            dom_l = dom.split("/")[-1].split("#")[-1] if dom else "Any"
            lines.append(f"  - {p_label}: on {dom_l} ({prop})")

        return "\n".join(lines)