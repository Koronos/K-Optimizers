Round 2 (B, first pass) was **discarded** per protocol step 5: total wall time 914.3s vs round 3's
(also B) 644.6s is a 41.8% deviation, over the 20% threshold. nvidia-smi (`../smi_campaign.csv`) shows
no sustained thermal/power throttling during this window (max temp 64C, only transient
sw_power_cap+hw_power_brake blips consistent with the rest of the campaign) -- so this looks like a
one-off host/OS scheduling artifact, not electrical. Consistent with this, `adapnm/unet/fused_stochastic_rounding`
alone swung 2.69ms (round 1, A) -> 9.29ms (this round) -> 1.68ms (round 3, B), a single-arm outlier not
reproduced elsewhere.

Round 2 was re-run as `../2r_B_026621d/` (609.9s, only 5.4% off round 3's 644.6s) and *that* replacement
is the one used in `../arms_table.md` and the final report. This directory's `battery_2_B_026621d.json`
is kept for transparency but excluded from all aggregates.
