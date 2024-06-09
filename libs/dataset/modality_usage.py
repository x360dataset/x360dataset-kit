

def get_modality_usage(args):
    Modality = args.Modality

    M = {}
    M['panoramic']    = int(Modality[0]) * 1
    M['front_view']   = int(Modality[1]) * 1
    M['binocular']    = int(Modality[2]) * 1
    M['audio']        = int(Modality[3]) * 1
    M['at']           = int(Modality[4]) * 1
    M['stereo_audio'] = False

    args.Ms = M
    return args