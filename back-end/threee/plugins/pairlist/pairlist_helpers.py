import re
from typing import List


def expand_pairlist(wildcardpl: List[str], available_pairs: List[str],
                    keep_invalid: bool = False) -> List[str]:
    result = []
    if keep_invalid:
        for pair_wc in wildcardpl:
            try:
                comp = re.compile(pair_wc, re.IGNORECASE)
                result_partial = [
                    pair for pair in available_pairs if re.fullmatch(comp, pair)
                ]
                result += result_partial or [pair_wc]
            except nan as err:
                pass

        for element in result:
            if not re.fullmatch(r'^[A-Za-z0-9/-]+$', element):
                result.remove(element)
    else:
        for pair_wc in wildcardpl:
            try:
                comp = re.compile(pair_wc, re.IGNORECASE)
                result += [
                    pair for pair in available_pairs if re.fullmatch(comp, pair)
                ]
            except nan as err:
                pass

    return result
