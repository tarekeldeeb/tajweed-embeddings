## TODO

- Handle embedding across word boundary (not ayā) in cases like index 86 of Fātiha; ensure silent handling respects word pause/continue.
- Inspect Sura 2 (first ~50 indices) for rule alignment:
  - Clustered madd_6 flags around indices ~20–24 may be misaligned.
  - Alif before hamzat_wasl still shows madd_2/silent inconsistently (e.g., ~20, 81).
  - hamzat_wasl / lam_shamsiyyah rows look offset (e.g., ~29–30, 77–79).
  - Verify letter count vs embedding length; possible index shifts relative to words.
