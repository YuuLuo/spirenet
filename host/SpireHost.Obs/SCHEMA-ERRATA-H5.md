# obs-schema-v1, erratum "H5": 2-4 seats served to one controller (TEAM view)

Author: the H5 host engineer. Date: 2026-11-15. Applies to STS2 v0.113.0, `SpireHost.Obs` of the `h5-4p` build.
Rank: `notes\obs-schema-v1.md` + `notes\schema-errata-v1.md`, amended by this file where they differ.
Approved by the lead as ruling Q7 of `notes\h5-plan-rulings.md`. C# constants: `host\SpireHost.Obs\Schema\SchemaV1.cs`
(`ResetFlag`, `Gl.SeatScreen0 / SeatStatus0`, `MapNodeFlag.VoteSeat0`, `OptionFlag.VoteSeat0`) and
`host\SpireHost.Driver\DecisionModel.cs` (`SeatStatus`).

**Nothing changes for a single-seat run.** No table, no column, no vocabulary entry is added; `SCHEMA_VERSION` stays 1;
`VOCAB_HASH` / `CATALOG_HASH` are unchanged. Every field below is 0 / absent in every frame of a single-seat episode,
and a single-seat RESET is accepted and refused exactly as before (including the refusal of flag bits 6-7).

## H5-1 RESET (amends P-2)

* `n_seats` 2, 3 and 4 are accepted with `view_mode = 0` (TEAM, `view_seat = -1`). A multi-seat RESET with
  `view_mode = 1` (SEAT) is answered with ERROR 3: the SEAT view is not built.
* `char0 .. char(n_seats-1)` are the seats' characters in seat order; the same character may sit on several seats.
  Seat s = index s of `RunState.Players`; seat 0 is the host player of the multiplayer run (no observable difference).
* `ascension` is clamped to the profile's multiplayer ascension (10), as the real lobby does.
* `flags` gets two bits that are defined **only when `n_seats > 1`** (with one seat they stay undefined: ERROR 3):

  | bit | name | 0 (default) | 1 |
  |---|---|---|---|
  | 6 | `ASK_EVERY_SEAT` | team votes: a vote whose result is one draw from all seats' votes (map vote, page vote of a shared event) is asked ONCE - from the lowest seat that has not voted, `acting_seat` = that seat - and the host casts the same vote for every other seat | every seat is asked for its own vote, in seat order |
  | 7 | `OFFER_UNDO_END_TURN` | `UNDO_END_TURN` is never offered | offered to a seat that ended its turn while a teammate has not (the real button) |

  Bits 8 and above stay undefined (ERROR 3). The debug word 15 (forced HP, only with `--allow-debug-options`) applies
  to every seat.

## H5-2 Order of decisions (amends schema section 8)

One controller, one decision at a time. In this priority:

1. A pending prompt (CARD_SELECT, CARD_REWARD, BUNDLE, CRYSTAL, CONFIRM, and seat 0's REWARD screens), last opened
   first. `acting_seat` = the prompt's seat. Several players' prompts can be open at once (e.g. end-of-turn prompts);
   they are asked one after the other.
2. A REWARD screen of a remote seat (seats 1..n-1), lowest seat first. `acting_seat` = that seat.
3. In combat: one COMBAT frame that is the union of all seats' legal actions. `acting_seat` = -1, or the one seat
   that has actions. Each ACTIONS row carries its seat.
4. Outside combat the room screens are asked seat by seat (seat-major): the lowest seat that has a decision on its
   own screen (EVENT / SHOP / REST / TREASURE) goes on until its PROCEED. `acting_seat` = that seat; OPTIONS / OFFER
   rows are that seat's own screen (`OPTIONS.seat` = `acting_seat`).
5. Team votes come last: the MAP frame and the EVENT frame of a shared-event page are asked only when no seat has any
   other decision. Default: once (H5-1 bit 6). The TREASURE relic vote is not a team vote: every seat is asked in its
   own chain (the seats vote for different relics) and sees the votes cast so far.

Consequences a client can rely on: out of combat `acting_seat` is never -1; a seat is never asked again for a vote it
has cast (no re-vote); the act change after a boss needs no extra decision (it is the PROCEED of each seat's boss
REWARD screen); the episode ends (WIN) when seat 0 presses the last button of the final event.

## H5-3 GLOBAL 52..59 (formerly reserved; amends section 4 and errata PY-1)

Written in every frame of a multi-seat episode (OBS and TERMINAL); 0 in single-seat frames; slots of seats
`>= n_seats` are 0.

| index | name | meaning |
|---|---|---|
| 52 + s | `seat_screen[s]` | the `DecisionKind` (schema A.1) of the screen seat s has open: for `acting_seat` of a single-seat decision the frame's own KIND; otherwise COMBAT (2) while a combat is in progress, REWARD (4) while that (remote) seat has a reward screen open, 0 once the seat is ready for the next act, MAP (1) once the seat left its room screen, else the room's screen EVENT (6) / SHOP (7) / REST (8) / TREASURE (9); 0 = none of these |
| 56 + s | `seat_status[s]` | bit 0 PROCEEDED (the seat left its room screen: its map is open); bit 1 VOTED_MAP; bit 2 ACT_READY (boss PROCEED pressed); bit 3 VOTED_SHARED_PAGE (current page of a shared event); bit 4 VOTED_TREASURE; bit 5 REST_DONE (a successful rest-site choice). All bits are cleared when a room is entered; bit 3 also when a page is resolved |

Everything here is public in the real client: the teammate panel shows each player's screen type, and every vote is
displayed. 60..63 stay reserved (0).

## H5-4 MAP_NODES.flags bits 4..7 (schema 5.10: now written)

Bit (4 + s) is set on the node seat s has voted for at the current location. Present in MAP frames of a multi-seat
episode when votes exist (i.e. with H5-1 bit 6: the frames of the seats that vote after the first one). Each seat's
vote is also visible as `seat_status[s]` bit 1.

## H5-5 OPTIONS.flags bits 7..10 (amends 5.11)

Bit (7 + s) = seat s voted for this option. Set on `TREASURE_RELIC` rows (RELIC page of a multi-seat TREASURE frame)
and on `EVENT_OPTION` rows of a shared event's page. Bits 0..6 are unchanged.

## H5-6 ACTIONS.aux for the rest option MEND (amends E-12)

MEND exists only with more than one player. It is offered as one action per target:
`CHOOSE_OPTION(seat = owner, src = the MEND row of OPTIONS, aux = target seat)`; the owner is never a target, so
`aux != seat`. For every other REST action `aux` stays 0 - a MEND action is recognised by its source row
(`rest_option` = `MEND`), not by `aux != 0` (target seat 0 has `aux = 0`). The canonical action order is unchanged
(verb, seat, src_table, src_row, tgt_row, aux).

## H5-7 What a TEAM frame shows of each seat (amends 2.3, 14.1)

TEAM view = the union of what the four players' own clients show; nothing is added that no player sees.

* Every seat, always: SEATS row (HP, block, gold, energy, counts), hand / piles in combat (piles as multisets, as for
  one seat), deck, relics, potions, orbs, powers with the per-seat visibility mask, intents with per-seat damage.
* A seat's reward list, card-reward cards, shop inventory, event page and options, rest options and crystal grid are
  encoded only in that seat's own decision frame, with the same gating as for one seat (no contents of an unopened
  card reward, no item under a hidden crystal cell, chest relics only after that seat opened its chest).
* Shared: map and map votes (H5-4), treasure relics and votes (H5-5), shared-event page and votes (H5-5), per-seat
  screen and status (H5-3).
* Never encoded: draw order, any RNG state or seed - including the three streams that resolve votes
  (`map_point_selection`, `event_synchronizer`, the treasure relic stream) -, unopened card rewards of any seat.

Verified by `obs --leak-test --seats N` (perturbations P1-P13 for every seat, P15 vote RNGs, P16 unopened card rewards
of all seats, P17 / P18 other seats' shop inventories / rest options; positive controls C1-C10).

## H5-8 Python-side changes this erratum asks for

| # | file | change |
|---|---|---|
| PY-H1 | `spire/schema.py` | `G.SEAT_SCREEN0..3 = 52..55`, `G.SEAT_STATUS0..3 = 56..59`; `class SeatStatus(IntFlag)` with the six bits of H5-3; `OptionFlag.VOTE_SEAT0..3 = 1 << 7 .. 1 << 10`; `ResetFlag.ASK_EVERY_SEAT = 1 << 6`, `OFFER_UNDO_END_TURN = 1 << 7`; docstring: "52..59 erratum H5, 60..63 reserved" |
| PY-H2 | `spire/proto.py` | `ResetCmd` may set bits 6 / 7 only when `n_seats > 1`; no other protocol change (RESET with 2-4 characters and TEAM view was already encodable) |
| PY-H3 | `spire/tokenize.py` + C kernel | SEAT token: screen kind (embedding) and status bits; OPTION token: 11 flag bits; ACTION token: `aux` as target-seat link when the source row is the MEND option |
| PY-H4 | caps | decisions per episode grow with the seat count (measured: `notes\host-h5-4p.md`); raise `max_episode_decisions` accordingly |
