"""Registered offline state positions; never a generator input or online feature."""


def quartile_ordinals(count):
    if type(count) is not int or count < 4:
        raise ValueError('Four distinct real decisions require at least four customers')
    return [(n * count + 7) // 8 - 1 for n in (1, 3, 5, 7)]


def capture_ordinals_for_day(collection, day):
    if day not in collection['days']:
        raise ValueError('Unregistered capture day')
    if 'capture_ordinals_by_day' in collection:
        if 'capture_ordinals' in collection:
            raise ValueError('Ambiguous global and per-day capture positions')
        return list(collection['capture_ordinals_by_day'][str(day)])
    return list(collection['capture_ordinals'])


def registered_capture_keys(collection):
    return [(day, ordinal) for day in collection['days']
            for ordinal in capture_ordinals_for_day(collection, day)]


def validate_capture_schedule(collection, actual_counts=None):
    days = collection['days']
    if not days or any(type(d) is not int or d < 0 for d in days) or len(set(days)) != len(days):
        raise ValueError('Explicit unique capture days required')
    mapped = 'capture_ordinals_by_day' in collection
    if mapped:
        if 'capture_ordinals' in collection:
            raise ValueError('Ambiguous global and per-day capture positions')
        design = collection['capture_design']
        if (design['rule'], design['numerators'], design['denominator']) != (
                'ceil(q*N_day)-1', [1, 3, 5, 7], 8):
            raise ValueError('Unsupported registered sampling design')
        expected = {str(day) for day in days}
        if (set(collection['capture_ordinals_by_day']) != expected or
                set(design['actual_orders_by_day']) != expected):
            raise ValueError('Missing or unexpected mapped capture day')
    for day in days:
        positions = capture_ordinals_for_day(collection, day)
        if (not positions or positions != sorted(positions) or len(set(positions)) != len(positions)
                or any(type(i) is not int or i < 0 or i >= collection['orders'] for i in positions)):
            raise ValueError('Missing, duplicated, unordered or out-of-range position')
        if mapped:
            count = collection['capture_design']['actual_orders_by_day'][str(day)]
            if type(count) is not int or count > collection['orders'] or positions != quartile_ordinals(count):
                raise ValueError('Capture positions differ from registered quartiles')
            if actual_counts is not None and int(actual_counts[day]) != count:
                raise ValueError('Actual regenerated customer count differs from registration')
        if actual_counts is not None and any(i >= int(actual_counts[day]) for i in positions):
            raise ValueError('Registered capture position does not exist on this day')
    return registered_capture_keys(collection)
