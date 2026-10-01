from typing import Dict, Any, List, Optional, Set, Tuple
import pipmaster as pm
from ascii_colors import ASCIIColors

try:
    import rdflib
    from rdflib import Graph, URIRef, Literal, RDF, RDFS, OWL
except ImportError:
    pm.ensure_packages(["rdflib"])
    import rdflib
    from rdflib import Graph, URIRef, Literal, RDF, RDFS, OWL


class SemanticReasoner:
    """
    Deductive RDFS and OWL Reasoning Engine for SafeStore Knowledge Graphs.

    Performs forward-chaining rule expansion and logical consistency validation:
    1. Transitive Subclass & Subproperty Inheritance (RDFS)
    2. Domain and Range Type Materialization (RDFS)
    3. Inverse Property Materialization (owl:inverseOf)
    4. Symmetric Property Reversal (owl:SymmetricProperty)
    5. Transitive Property Chaining (owl:TransitiveProperty)
    6. Equivalence Unification (owl:sameAs, owl:equivalentClass, owl:equivalentProperty)
    7. Disjointness & Exclusivity Verification (owl:disjointWith, owl:differentFrom, owl:FunctionalProperty)
    """

    def __init__(self, graph: Graph, tbox: Optional[Any] = None):
        self.graph = graph
        self.tbox = tbox

    def run_inference(self, max_iterations: int = 5) -> Tuple[int, List[Dict[str, str]]]:
        """
        Executes deductive forward-chaining closure over the graph.

        Returns:
            (inferred_triples_count, list_of_inferred_triples)
        """
        # If tbox is provided and has an ontology graph, merge it
        if self.tbox and hasattr(self.tbox, "graph"):
            for triple in self.tbox.graph:
                self.graph.add(triple)

        # Attempt to run owlrl if installed
        try:
            import owlrl
            initial_count = len(self.graph)
            owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(self.graph)
            inferred_count = len(self.graph) - initial_count
            ASCIIColors.info(f"OWLRL reasoning engine inferred {inferred_count} triples.")
        except Exception:
            pass

        # Apply SafeStore native deductive closure rules
        new_triples: Set[Tuple[Any, Any, Any]] = set()
        for _ in range(max_iterations):
            round_new: Set[Tuple[Any, Any, Any]] = set()

            # Rule 1: Subclass inheritance: if ?s a ?C and ?C rdfs:subClassOf ?D => ?s a ?D
            for s, _, c in list(self.graph.triples((None, RDF.type, None))):
                for _, _, d in self.graph.triples((c, RDFS.subClassOf, None)):
                    if d not in (OWL.Thing, RDFS.Resource) and (s, RDF.type, d) not in self.graph:
                        round_new.add((s, RDF.type, d))

            # Rule 2: Subproperty inheritance: if ?s ?p ?o and ?p rdfs:subPropertyOf ?q => ?s ?q ?o
            for s, p, o in list(self.graph):
                for _, _, q in self.graph.triples((p, RDFS.subPropertyOf, None)):
                    if (s, q, o) not in self.graph:
                        round_new.add((s, q, o))

            # Rule 3: Inverse properties: if ?s ?p ?o and ?p owl:inverseOf ?q => ?o ?q ?s
            for p, _, q in list(self.graph.triples((None, OWL.inverseOf, None))):
                for s, _, o in list(self.graph.triples((None, p, None))):
                    if (o, q, s) not in self.graph:
                        round_new.add((o, q, s))
                for s, _, o in list(self.graph.triples((None, q, None))):
                    if (o, p, s) not in self.graph:
                        round_new.add((o, p, s))

            # Rule 4: Symmetric properties: if ?p a owl:SymmetricProperty and ?s ?p ?o => ?o ?p ?s
            for p, _, _ in list(self.graph.triples((None, RDF.type, OWL.SymmetricProperty))):
                for s, _, o in list(self.graph.triples((None, p, None))):
                    if isinstance(o, URIRef) and (o, p, s) not in self.graph:
                        round_new.add((o, p, s))

            # Rule 5: Transitive properties: if ?p a owl:TransitiveProperty, ?x ?p ?y and ?y ?p ?z => ?x ?p ?z
            for p, _, _ in list(self.graph.triples((None, RDF.type, OWL.TransitiveProperty))):
                for x, _, y in list(self.graph.triples((None, p, None))):
                    if isinstance(y, URIRef):
                        for _, _, z in list(self.graph.triples((y, p, None))):
                            if (x, p, z) not in self.graph:
                                round_new.add((x, p, z))

            # Rule 6: Domain & Range typings
            for p, _, domain in list(self.graph.triples((None, RDFS.domain, None))):
                for s, _, _ in list(self.graph.triples((None, p, None))):
                    if (s, RDF.type, domain) not in self.graph:
                        round_new.add((s, RDF.type, domain))

            for p, _, range_ in list(self.graph.triples((None, RDFS.range, None))):
                for _, _, o in list(self.graph.triples((None, p, None))):
                    if isinstance(o, URIRef) and (o, RDF.type, range_) not in self.graph:
                        round_new.add((o, RDF.type, range_))

            # Rule 7: Equivalence propagation (owl:sameAs)
            for a, _, b in list(self.graph.triples((None, OWL.sameAs, None))):
                if isinstance(b, URIRef):
                    # Copy outgoing
                    for _, p, o in list(self.graph.triples((a, None, None))):
                        if p != OWL.sameAs and (b, p, o) not in self.graph:
                            round_new.add((b, p, o))
                    # Copy incoming
                    for s, p, _ in list(self.graph.triples((None, None, a))):
                        if p != OWL.sameAs and (s, p, b) not in self.graph:
                            round_new.add((s, p, b))

            if not round_new:
                break

            for triple in round_new:
                self.graph.add(triple)
                new_triples.add(triple)

        formatted_inferred = []
        for s, p, o in new_triples:
            formatted_inferred.append({
                "subject": str(s),
                "predicate": str(p),
                "object": str(o)
            })

        return len(new_triples), formatted_inferred

    def check_consistency(self) -> Dict[str, Any]:
        """
        Validates the knowledge graph against logical exclusivity constraints:
        - Disjoint classes (owl:disjointWith)
        - Inequality / Distinctness conflicts (owl:differentFrom)
        - Functional property cardinality violations (owl:FunctionalProperty)

        Returns:
            Dict containing boolean 'is_consistent' and list of 'violations'.
        """
        violations: List[Dict[str, Any]] = []

        # 1. Check Disjoint Classes Exclusivity (owl:disjointWith)
        disjoint_pairs: Set[Tuple[URIRef, URIRef]] = set()
        for c1, _, c2 in self.graph.triples((None, OWL.disjointWith, None)):
            disjoint_pairs.add((c1, c2))
            disjoint_pairs.add((c2, c1))

        # Check all subjects for dual membership in disjoint classes
        entity_types: Dict[Any, Set[Any]] = {}
        for s, _, t in self.graph.triples((None, RDF.type, None)):
            if s not in entity_types:
                entity_types[s] = set()
            entity_types[s].add(t)

        for entity, types in entity_types.items():
            for c1, c2 in disjoint_pairs:
                if c1 in types and c2 in types:
                    c1_name = str(c1).split('/')[-1].split('#')[-1]
                    c2_name = str(c2).split('/')[-1].split('#')[-1]
                    e_name = str(entity).split('/')[-1].split('#')[-1]
                    violations.append({
                        "type": "disjoint_class_violation",
                        "severity": "CRITICAL",
                        "entity": str(entity),
                        "entity_label": e_name,
                        "conflict": f"Entity '{e_name}' is declared as both '{c1_name}' and '{c2_name}', which are strictly disjoint (mutually exclusive)."
                    })

        # 2. Check Functional Properties Multi-Value Conflicts
        functional_props = set(s for s, _, _ in self.graph.triples((None, RDF.type, OWL.FunctionalProperty)))
        for prop in functional_props:
            subject_values: Dict[Any, Set[Any]] = {}
            for s, _, o in self.graph.triples((None, prop, None)):
                if s not in subject_values:
                    subject_values[s] = set()
                subject_values[s].add(str(o))

            for subj, vals in subject_values.items():
                if len(vals) > 1:
                    p_name = str(prop).split('/')[-1].split('#')[-1]
                    s_name = str(subj).split('/')[-1].split('#')[-1]
                    violations.append({
                        "type": "functional_property_violation",
                        "severity": "WARNING",
                        "entity": str(subj),
                        "property": str(prop),
                        "conflict": f"Functional property '{p_name}' has {len(vals)} distinct values for subject '{s_name}': {list(vals)} (expected at most 1)."
                    })

        # 3. Check Inequality Conflicts (owl:differentFrom vs owl:sameAs)
        for a, _, b in self.graph.triples((None, OWL.differentFrom, None)):
            if (a, OWL.sameAs, b) in self.graph or (b, OWL.sameAs, a) in self.graph:
                violations.append({
                    "type": "inequality_contradiction",
                    "severity": "CRITICAL",
                    "entity": str(a),
                    "conflict": f"Entities '{str(a).split('/')[-1]}' and '{str(b).split('/')[-1]}' are declared both owl:differentFrom and owl:sameAs."
                })

        return {
            "is_consistent": len(violations) == 0,
            "violation_count": len(violations),
            "violations": violations
        }