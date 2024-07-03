def get_modality_usage(args):
    modality = args.modality

    M = {}
    M['panoramic']    = int(modality[0]) * 1
    M['front_view']   = int(modality[1]) * 1
    M['binocular']    = int(modality[2]) * 1
    M['audio']        = int(modality[3]) * 1
    M['at']           = int(modality[4]) * 1
    M['stereo_audio'] = 0

    args.Ms = M
    return args