from typing import List, Dict
from objects import Trade

def time_filter(obj, arg) -> bool:
    return True

def trade_filter(new_trades: List[Trade], attributes: Dict[str, str]) -> bool:
    is_filter_passed = False
    for trade in new_trades:

        all_atr_matched = True
        for atr_path, atr_value in attributes.items():
            x = str(eval(atr_path))
            if str(eval(atr_path)) != atr_value:
                all_atr_matched = False
                break

        if all_atr_matched:
            is_filter_passed = True
            break

    return is_filter_passed