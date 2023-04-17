

# The 'new' system changed some of the 'old' camera view numbering, so we need to map the old channels to the new ones
ch_old_to_new = {
        '001': '003',
        '002': '006',
        '003': '002',
        '004': '001',
        '005': '005',
        '006': '004',
        '007': '007',
        '008': '008',
}
ch_new_to_old = {v: k for k, v in ch_old_to_new.items()}

# Camera capture system versions for subjects
cam_ver = {
        'S01': 'new', 'S02': 'old', 'S03': 'old', 'S04': 'old', 'S05': 'old',
        'S06': 'old', 'S07': 'old', 'S08': 'old', 'S09': 'old', 'S10': 'old',
        'S11': 'old', 'S12': 'old', 'S13': 'old', 'S14': 'old', 'S15': 'old',
        'S16': 'old', 'S17': 'old', 'S18': 'old', 'S19': 'old', 'S20': 'old',
        'S21': 'old', 'S22': 'old', 'S23': 'old', 'S24': 'old', 'S25': 'old',
        'S26': 'old', 'S27': 'old', 'S28': 'new', 'S29': 'new', 'S30': 'old',
        'S31': 'new', 'S32': 'old', 'S33': 'old', 'S34': 'old', 'S35': 'old',
}