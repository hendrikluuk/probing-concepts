import random
import traceback

def sample_subclasses(referents: dict, total:int=24, min_k:int=3, max_k:int=6, max_word_len:int=100) -> dict:
    """
        Sample equal number of referents from min_k to max_k subclasses of referents

        min_k: minimum number of subclasses to sample from
    """
    # eligible subclasses
    subclasses = get_subclasses(referents, total // min_k, max_word_len=max_word_len) 
    n = len(subclasses)
    if n < min_k:
        return {}

    # number of subclasses to sample from
    k = min_k
    for i in range(3, min(max_k+1, n+1)):
        if i == 5:
            continue
        if n % i == 0:
            k = i

    # number of referents to sample from each subclass
    m = total // k

    try:
        # sample k subclasses without replacement
        selection = random.sample(list(subclasses.keys()), k)
        subclasses = {key: subclasses[key] for key in selection}
    except Exception as e:
        print(f"Error in sampling subclasses from {list(subclasses.keys())}: {e}")
        traceback.print_exc()
        return {}

    # sample m referents from each subclass
    siblings = {}
    for key in subclasses:
        siblings[key] = random.sample(subclasses[key], m)

    # randomize the order of sampled referents
    entities = [entity for sublist in siblings.values() for entity in sublist]
    random.shuffle(entities)
    return {
        "entities": entities,
        "siblings": siblings
    }

def get_subclasses(referents: dict, n:int, max_word_len:int=100) -> dict:
    """
      Get the list of subclasses with at least n children.
    """
    result = {}
    for key, value in referents.items():
        children = get_children(value, max_word_len=max_word_len)
        if len(children) >= n:
            result[key] = children
    return result

def get_children(referents: dict|list[str], max_word_len:int=100) -> list[str]:
    """
      Get the list of children in the referents tree.

      max_word_len: skip entities with names longer than {max_word_len}
    """
    if isinstance(referents, list):
        # return as is if not a tree
        return referents

    result = []
    for key, value in referents.items():
        # skip keys that are too long (some chemical entities have names >800 characters)
        if len(key) < max_word_len:
            # include the intermediate node id-s too (not just leaf nodes)
            result.append(key)
        if isinstance(value, dict) and value:
            result.extend(get_children(value))
    return result
