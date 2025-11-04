from tabulate import tabulate
import os

# Define headers and data
world = int(os.getenv("WORLD_SIZE", 0))
# rank = int(os.getenv("LOCAL_RANK", 0))
headers = ["Model"]
for i in range(world):
    headers.append(f"rank-{i}")


def parse_device_id(env_var_name, default_value, rank):
    device_id = os.getenv(env_var_name)
    if device_id:
        device_id = [int(dev) for dev in device_id.split(",")]
    else:
        device_id = [default_value]
    for i in range(len(device_id)):
        device_id[i] = device_id[i] * world + rank
    return device_id


data = [
    ["TE-1"],
    ["TE-2"],
    ["TE-3"],
    ["VAE"],
    ["TR."],
]


max_device_id = -1

for i in range(world):
    te_1_device_id = parse_device_id("TE_1_DEVICE_ID", 0, i)
    te_2_device_id = parse_device_id("TE_2_DEVICE_ID", 0, i)
    te_3_device_id = parse_device_id("TE_3_DEVICE_ID", 0, i)
    vae_device_id = parse_device_id("VAE_DEVICE_ID", 0, i)
    transformer_device_id = parse_device_id("TRANSFORMER_DEVICE_ID", 1, i)

    max_device_id = max(te_1_device_id, te_2_device_id, te_3_device_id, vae_device_id, transformer_device_id, key=lambda x: max(x))
    
    device_strs = [
        ",".join(str(x) for x in te_1_device_id),
        ",".join(str(x) for x in te_2_device_id),
        ",".join(str(x) for x in te_3_device_id),
        ",".join(str(x) for x in vae_device_id),
        ",".join(str(x) for x in transformer_device_id),
    ]
    
    for j in range(len(data)):
        data[j].append(device_strs[j])


# Print the table
print(tabulate(data, headers=headers, tablefmt="grid"))

print(f"Total devices needed: {max(max_device_id) + 1}")

# Usage example:
# TE_1_DEVICE_ID=0,1 TE_2_DEVICE_ID=2,3 TE_3_DEVICE_ID=4,5 VAE_DEVICE_ID=6,7 TRANSFORMER_DEVICE_ID=8,9 WORLD_SIZE=5 python visualize_mapping.py