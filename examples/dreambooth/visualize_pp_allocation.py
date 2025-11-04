def map_layers_to_devices(layers, device_ids):
    total_layers = len(layers)
    num_devices = len(device_ids)

    base_chunk = total_layers // num_devices
    remainder = total_layers % num_devices

    layer_to_device = {}
    start = 0

    for i, device_id in enumerate(device_ids):
        extra = 1 if i < remainder else 0
        end = start + base_chunk + extra
        for layer in layers[start:end]:
            layer_to_device[layer] = device_id
        start = end

    return layer_to_device


text_encoder_three_layer_list = [
    'shared', 'encoder.embed_tokens',
    'encoder.block.0', 'encoder.block.1', 'encoder.block.2', 'encoder.block.3',
    'encoder.block.4', 'encoder.block.5', 'encoder.block.6', 'encoder.block.7',
    'encoder.block.8', 'encoder.block.9', 'encoder.block.10', 'encoder.block.11',
    'encoder.block.12', 'encoder.block.13', 'encoder.block.14', 'encoder.block.15',
    'encoder.block.16', 'encoder.block.17', 'encoder.block.18', 'encoder.block.19',
    'encoder.block.20', 'encoder.block.21', 'encoder.block.22', 'encoder.block.23',
    'encoder.final_layer_norm', 'encoder.dropout'
]

device_ids = [4, 10, 12]  # or [1, 2, 3, 4, 5]

allocation = map_layers_to_devices(text_encoder_three_layer_list, device_ids)
print(allocation)
# Print the allocation

for device, assigned_layers in allocation.items():
    print(f"Device {device}: {assigned_layers}")