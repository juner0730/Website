from typing import Optional, Tuple, List
from predict import detect_jersey_number

def detect(frame, bbox) -> Optional[Tuple[str, List[float]]]:
    return detect_jersey_number(frame, bbox)
