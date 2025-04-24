import os
import json
from collections import Counter

import numpy as np
import pandas as pd

from utils.subclasses import get_children

source_mapping = {
    "CHEBI": "https://www.ebi.ac.uk/chebi",
    "GO": "http://geneontology.org",
    "MeSH": "https://www.ncbi.nlm.nih.gov/mesh",
    "MedDRA": "https://www.meddra.org",
    "MOP": "https://www.ebi.ac.uk/ols/ontologies/mop",
    "Reactome": "https://reactome.org",
}

known_domains = set(["biology", "chemistry", "medicine"])

def concept_stats(concepts: list[dict], outfile: str = os.path.join("reports", "concept_stats.xlsx")):
    if os.path.exists(outfile):
        print(f"WARNING: {outfile} already exists. Will skip concept stats.")
        return

    # Print number of concepts with referents, with definitions, with selection criteria and total number of concepts and referents
    with_referents = 0
    with_definitions = 0
    with_selection_criteria = 0
    total_concepts = len(concepts)
    total_referents = []
    domain_counter = Counter()
    errors = []

    for concept in concepts:
        if concept.get("referents"):
            with_referents += 1
            total_referents.append(len(get_children(concept["referents"])))
        else:
            print(f"Concept '{concept['concept']}' has no referents")
            errors.append(f"no referents: '{concept['concept']}'")

        domain = concept.get("domain")
        if domain not in known_domains:
            print(f"WARNING: unknown domain '{domain}' listed for concept {concept['concept']}")
            errors.append(f"unknown domain: '{concept['concept']}'")

        if concept.get("definition"):
            with_definitions += 1
        else:
            print(f"WARNING: Concept '{concept['concept']}' has no definition")
            errors.append(f"no definition: '{concept['concept']}'")

        if concept.get("selection_criteria"):
            with_selection_criteria += 1
        else:
            print(f"WARNING: Concept '{concept['concept']}' has no selection criteria")
            errors.append(f"no selection criteria: '{concept['concept']}'")

    if errors:
        print(f"WARNING: found {len(errors)} errors:")
        for error in errors:
            print(error)

    print(f"Total concepts: {total_concepts}")
    print(f"Total referents: {sum(total_referents)}\n")
    print(f"Average number of referents per concept: {sum(total_referents)/total_concepts:.0f}")
    print(f"Minimum number of referents in concept: {min(total_referents)} ('{concepts[np.argmin(total_referents)]['concept']}')")
    print(f"Maximum number of referents in concept: {max(total_referents)} ('{concepts[np.argmax(total_referents)]['concept']}')\n")
    print(f"Concepts with referents: {with_referents}")
    print(f"Concepts with definitions: {with_definitions}")
    print(f"Concepts with selection criteria: {with_selection_criteria}")
    domains = {key: value for key, value in domain_counter.items()}
    print(f"Domains: {json.dumps(domains, indent=4)}")

    # Save the stats to an Excel file
    domains = ["Chemistry", "Biology", "Medicine", "Overall"]
    rows = ["Concepts", "Referents", "Referents per concept (average)", "Referents per concept (max)", "Referents per concept (min)"]
    data = {domain: [] for domain in domains}
    data["Overall"] = [total_concepts, sum(total_referents), round(sum(total_referents)/total_concepts), max(total_referents), min(total_referents)]

    for domain in domains[:-1]:
        domain_concepts = [concept for concept in concepts if concept.get("domain") == domain.lower()]
        domain_referents = [len(get_children(concept["referents"])) for concept in domain_concepts]
        data[domain] = [
            int(len(domain_concepts)),
            int(sum(domain_referents)),
            round(sum(domain_referents)/len(domain_concepts)),
            f"{int(max(domain_referents))} ({domain_concepts[np.argmax(domain_referents)]['concept']})",
            f"{int(min(domain_referents))} ({domain_concepts[np.argmin(domain_referents)]['concept']})"
        ]

    df = pd.DataFrame(data, index=rows)
    df = df.transpose()

    # make a table of concepts and ontologies that were used to gather information about the referents of concepts
    # relevant fields in concept: 'domain', 'ontology', 'referents_source', 'definition_source'
    # calculate the count of each ontology and domain name of referents_source
    sources = []
    for concept in concepts:
        source = {
            "concept": concept.get("concept"),
            "domain": concept.get("domain"),
            "ontology": concept.get("ontology"),
            "referents_source": concept.get("referents_source"),
            "definition_source": concept.get("definition_source")
        }
        sources.append(source)
    sources_df = pd.DataFrame(sources)
    # remove the last part of referents/definition source URL
    sources_df["referents_source"] = sources_df["referents_source"].apply(lambda x: "/".join(x.split("/")[:-1]) if x else "")
    sources_df["definition_source"] = sources_df["definition_source"].apply(lambda x: "/".join(x.split("/")[:-1]) if x else "")
    # concatenate value counts from ontology, referents_source and definition_source into one data series
    sources_df = pd.concat([
        sources_df["ontology"].value_counts(),
        sources_df["referents_source"].value_counts(),
        sources_df["definition_source"].value_counts()
    ], axis=0).sort_values(ascending=False)
    sources_df = sources_df.reset_index()
    sources_df.columns = ["Source", "Count"]
    sources_df["Source"] = [s if s not in source_mapping else source_mapping[s] for s in sources_df["Source"]]
    # merge counts on duplicate sources
    sources_df = sources_df.groupby("Source").sum().reset_index().sort_values(by="Count", ascending=False)

    with pd.ExcelWriter(outfile, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Concepts', index=False)
        sources_df.to_excel(writer, sheet_name='Sources')

    print(f"Concept stats saved to {outfile}")
    print(df)


