from pymol import cmd

cmd.reinitialize()
cmd.bg_color('white')
cmd.set('ray_trace_mode', 1)
cmd.set('ray_shadows', 0)
cmd.set('antialias', 2)

# Custom Colors
cmd.set_color('base_blue', [91/255, 150/255, 210/255])

# Load Structure
cmd.load('1PGA.cif', 'base_struct')
cmd.hide('everything', 'base_struct')
cmd.color('base_blue', 'base_struct')
cmd.show('cartoon', 'base_struct')

# Coloring Helper (Normalized)
def apply_spectrum_norm(obj_name, raw_max):
    cmd.color('base_blue', obj_name)
    print(f'Object {obj_name}: Raw Max = {raw_max:.4f}')
    if raw_max < 0.0001: return
    # Coloring only residues > 0.1 (10% max activation)
    selection = f'{obj_name} and b > 0.1'
    # White -> Red Spectrum
    cmd.spectrum('b', 'white_red', selection=selection, minimum=0.1, maximum=1.0)

# --- L0_248 (Raw Max: 10.9006) ---
cmd.create('L0_248', 'base_struct')
cmd.alter('L0_248', 'b=0.0')
cmd.alter('L0_248 and chain A and resi 24', 'b=1.0000')
cmd.alter('L0_248 and chain A and resi 43', 'b=0.9382')
apply_spectrum_norm('L0_248', 10.900619506835938)
cmd.group('Circuit_Analysis', 'L0_248')

# --- L2_3026 (Raw Max: 2.9974) ---
cmd.create('L2_3026', 'base_struct')
cmd.alter('L2_3026', 'b=0.0')
cmd.alter('L2_3026 and chain A and resi 3', 'b=0.5272')
cmd.alter('L2_3026 and chain A and resi 5', 'b=0.4123')
cmd.alter('L2_3026 and chain A and resi 41', 'b=0.4934')
cmd.alter('L2_3026 and chain A and resi 43', 'b=1.0000')
cmd.alter('L2_3026 and chain A and resi 52', 'b=0.6026')
apply_spectrum_norm('L2_3026', 2.997438430786133)
cmd.group('Circuit_Analysis', 'L2_3026')

# --- L5_3028 (Raw Max: 3.5918) ---
cmd.create('L5_3028', 'base_struct')
cmd.alter('L5_3028', 'b=0.0')
cmd.alter('L5_3028 and chain A and resi 16', 'b=0.6519')
cmd.alter('L5_3028 and chain A and resi 20', 'b=1.0000')
cmd.alter('L5_3028 and chain A and resi 21', 'b=0.9836')
apply_spectrum_norm('L5_3028', 3.591797351837158)
cmd.group('Circuit_Analysis', 'L5_3028')

# --- L5_1609 (Raw Max: 10.7820) ---
cmd.create('L5_1609', 'base_struct')
cmd.alter('L5_1609', 'b=0.0')
cmd.alter('L5_1609 and chain A and resi 24', 'b=1.0000')
cmd.alter('L5_1609 and chain A and resi 43', 'b=0.6909')
apply_spectrum_norm('L5_1609', 10.782029151916504)
cmd.group('Circuit_Analysis', 'L5_1609')

cmd.disable('base_struct')
cmd.disable('Circuit_Analysis')
cmd.zoom('base_struct')
print('Done! Enable specific objects in Circuit_Analysis to view.')