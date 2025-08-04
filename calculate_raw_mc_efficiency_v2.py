import uproot
import numpy as np

# ROOT file path
root_file_path = "mc_eff_hists_v2/effs_jpsi_resonant_L1_6p5_HLT_4p5.root"

# Trigger weights dictionary
trigger_weights = {
    "L1_11p0_HLT_6p5" : 4.66E-2,
    "L1_10p5_HLT_6p5" : 3.35E-2,
    "L1_10p5_HLT_5p0" : 3.04E-3,
    "L1_9p0_HLT_6p0"  : 2.61E-1,
    "L1_8p5_HLT_5p5"  : 9.86E-2,
    "L1_8p5_HLT_5p0"  : 1.99E-2,
    "L1_8p0_HLT_5p0"  : 2.04E-1,
    "L1_7p5_HLT_5p0"  : 4.83E-2,
    "L1_7p0_HLT_5p0"  : 7.86E-2,
    "L1_6p5_HLT_4p5"  : 1.07E-2,
    "L1_6p0_HLT_4p0"  : 7.42E-2,
    "L1_5p5_HLT_6p0"  : 4.42E-3,
    "L1_5p5_HLT_4p0"  : 1.91E-2,
    "L1_5p0_HLT_4p0"  : 1.22E-3,
    "L1_4p5_HLT_4p0"  : 8.85E-4,
}

# Open the ROOT file
with uproot.open(root_file_path) as file:
    hist_dir = file["hists"]
    total_efficiency = 0.0

    for trigger, weight in trigger_weights.items():
        num_key = f"hists/diel_m_{trigger}_num_ptbinned"
        denom_key = f"hists/diel_m_{trigger}_denom_ptbinned"

        try:
            num_hist = file[num_key]
            denom_hist = file[denom_key]

            num_sum = np.sum(num_hist.values())
            denom_sum = np.sum(denom_hist.values())

            efficiency = num_sum / denom_sum if denom_sum > 0 else 0.0
            weighted_eff = weight * efficiency
            total_efficiency += weighted_eff

            print(f"{trigger:<20} efficiency = {efficiency:.4f}, weighted = {weighted_eff:.4f}")

        except KeyError as e:
            print(f" Missing histogram for {trigger}: {e}")

    print(f"\nTotal weighted efficiency: {total_efficiency:.4f}")

