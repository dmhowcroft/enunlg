import logging

import box

logger = logging.getLogger(__name__)


class SlotValueMR(box.Box):
    def __init__(self, *args, **kwargs):
        super(SlotValueMR, self).__init__(*args, **kwargs)

    def __repr__(self):
        slot_value_pairs = ", ".join([f"{key}='{self[key]}'" for key in self.keys()])
        return f"{self.__class__.__name__}({slot_value_pairs})"

    def __str__(self):
        return self.__repr__()

    def as_frozen(self) -> "SlotValueMR":
        return SlotValueMR(self, frozen_box=True)

    def delex_slot(self, slot: str) -> None:
        """`slot` will be replaced by f"__{slot.upper()}__" if it is present in the MR"""
        if slot in self:
            self[slot] = f"__{slot.upper()}__"

    def can_delex(self, slot) -> bool:
        """Checks that `slot` is a key in this MR"""
        return slot in self


class MultivaluedSlotValueMR(box.Box):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, default_box=True, default_box_attr=list, **kwargs)

    def __repr__(self):
        slot_value_pairs = ", ".join([f"{key}='{self[key]}'" for key in self.keys()])
        return f"{self.__class__.__name__}({slot_value_pairs})"

    def __str__(self):
        return self.__repr__()