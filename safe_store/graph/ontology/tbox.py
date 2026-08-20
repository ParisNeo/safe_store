from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Set
import pipmaster as pm
from ascii_colors import ASCIIColors
from ...core.exceptions import ConfigurationError

try:
    import rdflib
    from rdflib import Graph, URIRef, RDF, RDFS, OWL, Literal
except ImportError:
    pm.ensure_packages(["rdflib"])
    import rdflib
    from rdflib import Graph, URIRef, RDF, RDFS, OWL, Literal


class TBoxManager:
    """
    Manages the Terminological Component (TBox) / Ontology schema of a knowledge base.
    Provides class hierarchy resolution, domain/range constraints, and prompt schema serialization.
    """

    def __init__(self, base_uri: str = "http://example.org/ontology/"):
        self.base_uri = base_uri
        self.graph = Graph()
        self._classes: Set[str] = set()
        self._object_properties: Set[str] = set()
        self._datatype_properties: Set[str] = set()

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