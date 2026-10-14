# Delivery actual-loaded-closure identity semantics

## What the round-20 candidate covers

- the running interpreter
- every explicitly listed input file
- every loaded Python module __file__ and __cached__
- every memory-mapped native path
- the module map, native map and explicit-path list

## What it does not cover

- files in the registered tree that the request never loads
- package trees that are not imported
- a lazy import that happens after the last capture
- semantic behaviour of a file whose bytes are unchanged

## What the old whole-tree gate covers instead

- every file in the registered tree whether or not it is loaded
- mutation of a file the request never loads

## Why the old gate cannot sit inside the request

- cost: a full tree scan measured 5.618649 s for 8903 files, which cannot fit inside an 8 s request together with inference and cold-chain evaluation

## Resolution registered for the new request

the request uses the candidate at three capture points inside the budget and states explicitly that whole-tree coverage is NOT provided inside the request; a whole-tree check remains an out-of-request publication precondition, run before the timed request, never inside it

## Capture points

- P1 after model and explicit files are loaded
- P2 after the first JAX call and the first native (C++) call
- P3 immediately before publication

## Fail-closed rule

publication is refused if any capture differs from the previous capture in file hashes, file set, module map, native map or explicit-path list; on refusal the original plan is retained, the request is recorded as partial and nothing is published
