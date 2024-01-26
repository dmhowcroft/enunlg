import collections
import logging
import random
import re

import torch

RegexRule = collections.namedtuple('RegexRule', ("match_expression", "replacement_expression"))


def count_parameters(model, log_table=True, print_table=False):
    """
    Based on https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model,
    forwarded to me by Jonas Groschwitz
    """
    from prettytable import PrettyTable
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    if log_table:
        logging.info(table)
        logging.info(f"Total Trainable Params: {total_params}")
    if print_table:
        print(table)
        print(f"Total Trainable Params: {total_params}")
    return total_params


def log_list_of_tensors_sizes(list_of_tensors, level=logging.DEBUG) -> None:
    logging.log(level, f"{len(list_of_tensors)=}")
    for task in list_of_tensors:
        logging.log(level, f"{task.size()}")


def log_sequence(seq, indent="") -> None:
    for element in seq:
        logging.info(f"{indent}{element}")


def set_random_seeds(seed) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


# ----
# BEGIN code copied over from e2e-cleaning -- TODO dependency?
# ----

REALIZATIONS = {
    'area': {
        'city centre': [
            '(?:city|town) cent(?:re|er)',
            'cent(?:re|er) of (?:the )?(?:city|town)',
            'in the cent(?:re|er)',
        ],
        'riverside': [
            'riverside',
            '(?:near|by|at|close to|along|on|off|beside) the river',
        ],
    },
    'eat_type': {
        'coffee shop': [
            'coffee[- ]+shop',
            'caf[eé]',
            'coffee',
        ],
        'pub': [
            'pub',
        ],
        'restaurant': [
            'restaurant',
        ],
    },
    'family_friendly': {
        'no': [
            r'(?:isn\'t|not|non|no)[ -]+(?:\w+ ){0,2}(?:child|children|family|kids|kid)[ -]+(?:friendly|orien(?:ta)?ted)',
            '(?:child|children|family|kids|kid)[ -]+unfriendly',
            'adults?[ -]+only',
            'only for adults',
            '(?:no|not) (?:kids|children|famil(?:y|ies))',
            '(?:not|no)(?: good| suitable| friendly| orien(?:ta)?ted| open(?:ed))? (?:at|for|to|with)(?: the)? (?:kids|children|family|families|all age)',
            '(?:kids?|child(?:ren)?|famil(?:y|ies)) (?:are|is)(?:n\'t| not) (?:welcome|allowed|accepted)',
            r'(?:does not|doesn\'t) (?:welcome|allow|accept) (?:\w+ ){0,2}(?:kids?|child(?:ren)?|famil(?:y|ies)|all age)',
            'adult (?:establishment|venue|place|establish)',
        ],
        'yes': [
            'for (?:kids|children|family|families)',
            'family place',
            'place to bring the(?: whole)? family',
            '(?:friendly|suitable|good|orien(?:ta)?ted|open(?:ed)) (?:at|with|to|for)(?: the)(?:kids?|child(?:ren)?|famil(?:y|ies)?|all age)',
            '(?:child|children|family|kids|kid)[ -]+(?:friendly|orien(?:ta)?ted)',
            '(?:kids?|child(?:ren)?|famil(?:y|ies)) (?:are|is) (?:welcome|allowed|accepted)',
            r'(?:welcomes?|allows?|accepts?) (?:\w+ ){0,2}(?:kids?|child(?:ren)?|famil(?:y|ies)|all age)',
        ],
    },
    'food': {
        'Chinese': ['Chinese', 'Chines'],
        'English': ['English', 'British'],
        'Fast food': ['Fast food'],
        'French': ['French'],
        'Indian': ['Indian'],
        'Italian': ['Italian'],
        'Japanese': ['Japanese'],
    },
    'name': [
        'Alimentum',
        'Aromi',
        'Bibimbap House',
        'Blue Spice',
        'Browns Cambridge',
        'Clowns',
        'Cocum',
        'Cotto',
        'Fitzbillies',
        'Giraffe',
        'Green Man',
        'Loch Fyne',
        'Midsummer House',
        'Strada',
        'Taste of Cambridge',
        'The Cambridge Blue',
        'The Cricketers',
        'The Dumpling Tree',
        'The Eagle',
        'The Golden Curry',
        'The Golden Palace',
        'The Mill',
        'The Olive Grove',
        'The Phoenix',
        'The Plough',
        'The Punter',
        'The Rice Boat',
        'The Twenty Two',
        'The Vaults',
        'The Waterman',
        'The Wrestlers',
        'Travellers Rest Beefeater',
        'Wildwood',
        'Zizzi',
        'X-name',
    ],
    'near': [
        'All Bar One',
        'Avalon',
        'Burger King',
        'Café Adriatic',
        'Café Brazil',
        'Café Rouge',
        'Café Sicilia',
        'Clare Hall',
        'Crowne Plaza Hotel',
        'Express by Holiday Inn',
        'Rainbow Vegetarian Café',
        'Raja Indian Cuisine',
        'Ranch',
        'The Bakers',
        'The Portland Arms',
        'The Rice Boat',
        'The Six Bells',
        'The Sorrento',
        'Yippee Noodle Bar',
        'X-near',
    ],
    'price_range': {
        "cheap": [
            "(?:inexpensive|cheap)(?:ly)?",
            "low[- ]+price[ds]?",
            "affordabl[ey]",
            r"prices?(?: range)?(?: \w+){0,3} low",
        ],
        "less than £20": [
            "(?:inexpensive|cheap)(?:ly)?",
            "affordabl[ey]",
            "(?:less than|under) £? *20",
            "moderately priced",  # Overlaps with moderate
            "low[- ]+price[ds]?",
            r"prices?(?: range)?(?: \w+){0,3} low",
        ],
        "more than £30": [
            "(?:more than|over) £? *30",
            "high[- ]+price[ds]?",  # Overlaps with high
            "expensive",
            "not cheap",
            r"prices?(?: range)?(?: \w+){0,3} high",
        ],
        "high": [
            "high[- ]+price[ds]?",
            "expensive",
            r"prices?(?: range)?(?: \w+){0,3} high",
        ],
        "moderate": [
            "(?:moderate|reasonable|ok|average)(?:ly)?[- ]+price[ds]?",
            "not cheap",
            "affordable",
            "mid[- ]+(?:range[- ]+)price[ds]?",
            r"prices?(?: range)?(?: \w+){0,3} (?:ok|average|moderate|reasonable)",
        ],
        "£20-25": [
            "£? *20 *(?:[-–]*|to) *25",
            "(?:moderate|reasonable|ok|average)(?:ly)?[- ]+price[ds]?",  # Overlaps with moderate
            r"prices?(?: range)?(?: \w+){0,3} (?:ok|average|moderate|reasonable)",  # Overlaps with moderate
            "affordable",
        ]
    },
    'rating': {
        "1 out of 5": [
            "(?:1|one)(?:(?: out)? of (?:5|five)(?: stars?)?|[- ]+stars?)",
            r"(?:rat(?:ings?|e[ds]?)|reviews?|standards?|quality)(?: \w+){0,2} (?:as )?(?:low|bad|poor)(?:(?: \w+){0,3} (?:1|one)(?:(?: out)? of (?:5|five)|[- ]+stars?))?",
            r"(?:low|bad|poor|(?:not|doesn't|isn't)(?: \w+){0,2} (:?good|well))(?:ly)?(?:[ -]+\w+){0,2}[ -]+(?:rat(?:ings?|ed)|reviews?|standards?|quality)(?:(?: \w+){0,3} (?:1|one)(?:(?: out)? of (?:5|five)(?: stars?)?|[- ]+stars?))?",
        ],
        "3 out of 5": [
            "(?:3|three)(?:(?: out)? of (?:5|five)(?: stars?)?|[- ]+stars?)",
            r"(?:rat(?:ings?|e[ds]?)|reviews?|standards?|quality)(?: \w+){0,2} (?:as )?average(?:(?: \w+){0,3} (?:3|three)(?:(?: out)? of (?:5|five)|[- ]+stars?))?",
            r"(?:average|(?<!very )(?:good|well))(?:ly)?(?:[ -]+\w+){0,2}[ -]+(?:rat(?:ings?|ed)|reviews?|standards?|quality)(?:(?: \w+){0,3} (?:3|three)(?:(?: out)? of (?:5|five)(?: stars?)?|[- ]+stars?))?",
        ],
        "5 out of 5": [
            "(?:5|five)(?:(?: out)? of (?:5|five)(?: stars?)?|[- ]+stars?)",
            r"(?:rat(?:ings?|e[ds]?)|reviews?|standards?|quality)(?: \w+){0,2} (?:as )?high(?:(?: \w+){0,3} (?:5|five)(?:(?: out)? of (?:5|five)|[- ]+stars?))?",
            r"(?:high|excellent|very good|great)(?:ly)?(?:[ -]+\w+){0,2}[ -]+(?:rat(?:ings?|ed)|reviews?|standards?|quality)(?:(?: \w+){0,3} (?:5|five)(?:(?: out)? of (?:5|five)(?: stars?)?|[- ]+stars?))?",
        ],
        "high": [
            "(?:5|five)(?:(?: out)? of (?:5|five)(?: stars?)?|[- ]+stars?)",
            r"(?:rat(?:ings?|e[ds]?)|reviews?|standards?|quality)(?: \w+){0,2} (?:as )?high",
            r"(?:high|excellent|very good|great|well)(?:ly)?(?:[ -]+\w+){0,2}[ -]+(?:rat(?:ings?|ed)|reviews?|standards?|quality)",
        ],
        "average": [
            "(?:3|three)(?:(?: out)? of (?:5|five)(?: stars?)?|[- ]+stars?)",
            r"(?:rat(?:ings?|e[ds]?)|reviews?|standards?|quality)(?: \w+){0,2} (?:as )?average",
            r"(?:average|(?<!very )(?:good|well))(?:ly)?(?:[ -]+\w+){0,2}[ -]+(?:rat(?:ings?|ed)|reviews?|standards?|quality)",
        ],
        "low": [
            "(?:1|one)(?:(?: out)? of (?:5|five)(?: stars?)?|[- ]+stars?)",
            r"(?:rat(?:ings?|e[ds]?)|reviews?|standards?|quality)(?: \w+){0,2} (?:as )?(?:low|bad|poor)",
            r"(?:low|bad|poor|(?:not|doesn't|isn't)(?: \w+){0,2} (?:well|good))(?:ly)?(?:[ -]+\w+){0,2}[ -]+(?:rat(?:ings?|ed)|reviews?|standards?|quality)",
        ],
    },
}


def compile_patterns(patterns):
    """Compile a list of patterns into one big option regex. Note that all of them will match whole words only."""
    # pad intent patterns with \b (word boundary), unless they contain '^'/'$' (start/end)
    return re.compile('|'.join([((r'\b' if not pat.startswith('^') else '') + pat +
                                 (r'\b' if not pat.endswith('$') else ''))
                                for pat in patterns]),
                      re.I | re.UNICODE)


# store "proper" capitalization of the values
CAPITALIZE = {}
# compile realization patterns
for slot in REALIZATIONS.keys():
    if isinstance(REALIZATIONS[slot], list):
        CAPITALIZE[slot] = {val.lower(): val for val in REALIZATIONS[slot]}
        REALIZATIONS[slot] = compile_patterns(REALIZATIONS[slot])
    else:
        CAPITALIZE[slot] = {val.lower(): val for val in REALIZATIONS[slot].keys()}
        for value in REALIZATIONS[slot].keys():
            REALIZATIONS[slot][value] = compile_patterns(REALIZATIONS[slot][value])


class Match(object):
    """Realization pattern match in the system output"""

    def __init__(self, slot, value, regex_match):
        self.slot = slot
        self.value = value
        self._start = regex_match.start()
        self._end = regex_match.end()

    def is_same_string(self, other):
        return (self._start == other._start and self._end == other._end)

    def is_substring(self, other):
        return ((self._start > other._start and self._end <= other._end) or
                (self._start >= other._start and self._end < other._end))

    def __eq__(self, other):
        return (self.slot == other.slot and self.value == other.value and self.is_same_string(other))

    def __str__(self):
        return 'Match[%s=%s:%d-%d]' % (self.slot, self.value, self._start, self._end)

    def __repr__(self):
        return str(self)


def regex_extract_e2e_mr(ref):
    """Classify the MR given a text. Returns a dict-based MR format
    for the system output MR and the gold-standard MR."""
    # create MR dict representation of the output text
    # first, collect all value matches
    matches = []
    for slot in REALIZATIONS.keys():
        # verbatim slot
        if not isinstance(REALIZATIONS[slot], dict):
            matches.extend([Match(slot, CAPITALIZE[slot][match.group(0).lower()], match)
                            for match in REALIZATIONS[slot].finditer(ref)])
        # slot with variable realizations
        else:
            # collect all matches for all values
            for value in REALIZATIONS[slot].keys():
                matches.extend([Match(slot, CAPITALIZE[slot][value.lower()], match)
                                for match in REALIZATIONS[slot][value].finditer(ref)])

    # then filter out those that are substrings/duplicates (let only one value match,
    # preferrably the one indicated by the true MR -- check with the MR dict)
    filt_matches = []
    for match in matches:
        skip = False
        for other_match in matches:
            if match is other_match:
                continue
            if (match.is_substring(other_match) or
                (match.is_same_string(other_match) and
                 other_match in filt_matches)):
                skip = True
                break
        if not skip:
            filt_matches.append(match)

    # now put it all into a dict
    out_dict = {}
    for match in filt_matches:
        out_dict[match.slot] = out_dict.get(match.slot, {})
        out_dict[match.slot][match.value] = out_dict[match.slot].get(value, 0) + 1

    return out_dict
