p = r'D:\PyCharm_\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem\scripts\tests\d3_l1_v118_synthetic_worker.py'
s = open(p, encoding='utf-8').read()

hold_block = """        if args.mode == 'one_slot_then_hold' and index == 1:
            # READY was already published, so the initialization cap must no longer apply.  The
            # sleep happens between two COMPLETED slots (slot 0 is already 'slot_finished' and slot
            # 1 has not been declared open yet), so no slot deadline is involved and the only
            # deadline that could still fire is the BATCH cap -- which is far away.  A supervisor
            # keyed to a fixed READY.json (which this worker never writes) would instead kill this
            # at the initialization deadline.
            time.sleep(float(os.environ.get('V119_HOLD_S', '20')))
"""
assert s.count(hold_block) == 1, s.count(hold_block)
s = s.replace(hold_block, "")

anchor = """        day, pos = int(row['day_id']), int(row['position'])
        jr.publish('slot', day_id=day, position=pos)
"""
assert s.count(anchor) == 1, s.count(anchor)
new_anchor = """        day, pos = int(row['day_id']), int(row['position'])
        if args.mode == 'one_slot_then_hold' and index == 1:
            # READY was already published, so the initialization cap must no longer apply.  This
            # sleep happens BETWEEN two completed slots: slot 0 is already 'slot_finished' and slot
            # 1 has NOT been declared open, so no slot deadline is involved.  The only deadline that
            # could still fire is the batch cap, which is far away.  A supervisor keyed to a fixed
            # READY.json (which this worker never writes) would instead kill this at the
            # initialization deadline.
            time.sleep(float(os.environ.get('V119_HOLD_S', '20')))
        jr.publish('slot', day_id=day, position=pos)
"""
s = s.replace(anchor, new_anchor)
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('hold moved before the slot declaration')
