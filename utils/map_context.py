
from utils.subclasses import sample_subclasses

def map_context(call: dict, context_map:dict, concept:dict) -> dict:
    # map concept to context
    context = {}
    for key, value in context_map.items():
        if not value:
            if key in concept:
                context[key] = concept[key]
            elif key == "entities" and call == "decide-referents":
                if not isinstance(concept.get("referents"), dict) or len(concept["referents"]) < 3:
                    # skip the test if there are no subclasses
                    return {}
                # sample equal number of referents from 3,4 or 6 subclasses of referents
                # total number of samples
                n = context_map.get("n", 24)
                siblings = sample_subclasses(concept["referents"], n)
                if not siblings:
                    # skip the test if there are not enough subclasses
                    return {}
                context.update(siblings)
        else: 
            context[key] = value
    return context