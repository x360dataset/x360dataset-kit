import os
from import_ import *
import shutil

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import sys

sys.path.append('/bask/projects/j/jiaoj-3d-vision/Hao/360x/360x_Video_Experiments')
from db_utils import database

db = database(check_data=False)


def inference(args, pr, net, criterion, data_set, data_loader, device='cuda', video_idx=None, video_name=None):
    # import pdb; pdb.set_trace()
    net.eval()
    img_path_list = []
    crw_itds = []
    baseline_itds = []

    mp4name = video_name.split('/')[-1]
    save_path = video_name.replace(mp4name, "at.npy")

    if not args.force:
        if os.path.exists(save_path):
            # print("PASS: ", save_path, " is exists")
            return

    # Length
    print("=== dataset length:", len(data_set))
    if len(data_set) == 0:
        print("=== empty dataset")
        return

    print("=== save to:", save_path)
    with torch.no_grad():
        for step, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Inference"):

            # import pdb; pdb.set_trace()
            img_paths = batch['img_path']
            audio = batch['audio']
            audio_rate = batch['audio_rate']
            delay_time = batch['delay_time'].to(device)



            out = predict(args, pr, net, batch, device)

            aff_L2R = criterion.inference(out, softmax=False)

            crw_itd = estimate_crw_itd(args, pr, aff_L2R, delay_time).detach().cpu()
            crw_itds.append(crw_itd)  # important  #

            for i in range(aff_L2R.shape[0]):

                img_path_list.append(img_paths[i])
                curr_audio = audio[i]

                if args.no_baseline:
                    baseline_itd = 0
                else:
                    baseline_itd = gcc_phat(args, pr, curr_audio, fs=audio_rate[i].item(), max_tau=pr.max_delay,
                                            interp=1)

                baseline_itds.append(baseline_itd)

    img_path_list = np.array(img_path_list)
    crw_itds = torch.cat(crw_itds, dim=-1).data.cpu().numpy() * 1000
    baseline_itds = np.array(baseline_itds) * 1000

    crw_itds = np.asarray(crw_itds)

    for i in range(img_path_list.shape[0]):
        crw_itd = - crw_itds[: i + 1]

        res = [crw_itd]
        bins = np.arange(i + 1) / data_set.meta_dict['frame_rate']

    print("save to:", save_path)
    np.save(save_path, crw_itds)

    if args.save_video:
        print("save video...")
        save_path = os.path.join(x360_path, 'vis')
        visualization(args, pr, data_set, data_loader, img_path_list, crw_itds, baseline_itds, video_idx, save_path)


def visualization(args, pr, data_set, data_loader, img_path_list, crw_itds, baseline_itds, video_idx, save_path):
    # import pdb; pdb.set_trace()
    #
    frame_folder = os.path.join(save_path, 'frames', f'video-{str(video_idx).zfill(3)}')
    video_folder = os.path.join(save_path, '0_videos')
    audio_folder = os.path.join(save_path, '1_audio')

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(frame_folder, exist_ok=True)
    os.makedirs(video_folder, exist_ok=True)
    os.makedirs(audio_folder, exist_ok=True)

    # audio part
    audio = data_set.audio
    audio_rate = data_set.audio_rate
    audio_name = os.path.join(audio_folder, f'audio-{str(video_idx).zfill(3)}.wav')
    audio = audio.transpose(0, 1).data.cpu().numpy()
    sf.write(audio_name, audio, audio_rate)

    # frame part
    frame_rate = data_set.meta_dict['frame_rate']
    for i in range(img_path_list.shape[0]):
        frame_name = os.path.join(frame_folder, f'frame-{str(i).zfill(3)}.jpg')
        frame_vis(args, pr, img_path_list, crw_itds, baseline_itds, frame_rate, i, frame_name)

    # Generate videos
    video_name = os.path.join(video_folder, f'video-{str(video_idx).zfill(3)}.mp4')
    print("video_name:", video_name)

    os.system(
        f"ffmpeg -v quiet -y -framerate {frame_rate} -i \"{frame_folder}\"/frame-%03d.jpg -i {audio_name} -vf \"crop=trunc(iw/2)*2:trunc(ih/2)*2\" -vcodec h264 -strict -2 -acodec aac -shortest {video_name}")
    os.system("rm -rf {}".format(audio_folder))
    os.system("rm -rf {}".format(os.path.join(save_path, 'frames')))


def frame_vis(args, pr, img_path_list, crw_itds, baseline_itds, frame_rate, i, frame_name):
    # import pdb; pdb.set_trace()
    sns.set_style('white')
    img_path = img_path_list[i]
    crw_itd = - crw_itds[: i + 1]
    baseline_itd = - baseline_itds[: i + 1]

    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, dpi=150)  # , figsize=(12, 7)

    # create image
    img = imageio.imread(img_path)  # frame img
    ax0.imshow(img)
    ax0.axis('off')

    # create plot
    bins = np.arange(i + 1) / frame_rate
    if args.no_baseline:
        res = [crw_itd]
    else:
        res = [crw_itd, baseline_itd]

    names = ['Ours', 'GCC-PHAT']
    markers = ['o', '*']
    for idx in range(len(res)):
        ax1.plot(bins, res[idx], label=names[idx], marker=markers[idx], linewidth=1.5, markersize=2)

    full_bins = np.arange(img_path_list.shape[0]) / frame_rate
    ax1.plot(full_bins, [0] * img_path_list.shape[0], linewidth=1, markersize=0, linestyle='dashed', color='grey')

    ax1.legend(loc='upper center', ncol=2, frameon=False, fontsize=8)
    ax1.set_xlabel('Time (s)', fontsize=8)
    # ax1.set_xticks([])
    ax1.set_ylabel('Time delay (ms)', fontsize=8, rotation='vertical')
    ax1.set_xlim([0, int(img_path_list.shape[0] / frame_rate)])
    ax1.set_ylim([-1.0, 1.0])
    ax1.set_yticks([-1.0, 0, 1.0])
    ax1.tick_params(axis='x', labelsize=6)
    ax1.tick_params(axis='y', labelsize=6)
    ax1.margins(x=0)

    ax2 = ax1.twinx()
    ax2.set_ylabel('L                R', fontsize=8)
    ax2.set_ylim([-1.0, 1.0])
    ax2.set_yticks([-1.0, 0, 1.0])
    ax2.tick_params(axis='y', labelsize=6)

    plt.savefig(frame_name, bbox_inches='tight')
    plt.close()


def test(args, device):
    # save dir
    gpus = torch.cuda.device_count()
    gpu_ids = list(range(gpus))

    # ----- get parameters for audio ----- #
    fn = getattr(params, args.setting)
    pr = fn()
    pr.dataloader = 'SingleVideoDataset'  # SingleVideoDataset
    update_param(args, pr)

    # ----- make dirs for results ----- #
    sys.stdout = utils.LoggerOutput(os.path.join('results', args.exp, 'log.txt'))
    os.makedirs('./results/' + args.exp, exist_ok=True)

    # ------------------------------------- #
    tqdm.write('{}'.format(args))
    tqdm.write('{}'.format(pr))
    # ------------------------------------ #

    # ----- Network ----- #
    net = models.__dict__[pr.net](args, pr, device=device).to(device)
    criterion = models.__dict__[pr.loss](args, pr, device)

    # -------- Loading checkpoints weights ------------- #
    if args.resume:
        resume = args.resume  # './checkpoints/' +
        net, _ = torch_utils.load_model(resume, net, device=device, strict=False)

    # ------------------- #
    net = nn.DataParallel(net, device_ids=gpu_ids)

    if isinstance(pr.list_vis, str):
        samples = []

        # csv_file = csv.DictReader(open(pr.list_vis, 'r'), delimiter=',')
        idlist = db.get_idlist()

        for id in idlist:
            front_view = db.get_data_from_id(id, get_nested_data=True)['front_view']
            # front_ = []
            # for video_path in front_view:
            #     video_name = video_path.split("/")[-1]
            #
            #     audio_path = list_sample.replace(".mp4", ".wav")
            #     frame_path = list_sample.replace(video_name, "frames")
            samples.extend(front_view)
            for i in front_view:
                print(i)

    print("TOTAL:", len(samples))

    if args.max_sample > 0:
        samples = samples[: args.max_sample]

    # Reverse
    samples = samples[::-1]


    #  --------- Testing ------------ #
    for i in tqdm(range(len(samples)), desc="Generating Video"):
        pr.list_test = samples[i]
        try:
           # ----- Dataset and Dataloader ----- #
            test_dataset, test_loader = torch_utils.get_dataloader(args, pr,
                                                               split='test',
                                                               shuffle=False,
                                                               drop_last=False)
        except:
            print("VACANT:", samples[i])
            continue

        inference(args, pr, net, criterion, test_dataset, test_loader, device, video_idx=i, video_name=samples[i])

        # try:
        #     --------------------------------- #
            # inference(args, pr, net, criterion, test_dataset, test_loader, device, video_idx=i, video_name=samples[i])
        # except:
        #     print("ERROR:", samples[i])
        #     continue



if __name__ == '__main__':
    parser = init_args(return_parser=True)

    parser.add_argument('--list_vis', type=str, default=None, required=False)
    parser.add_argument('--save_video', type=bool, default=False, required=False)
    parser.add_argument('--force', action='store_true')

    args = parser.parse_args()

    test(args, DEVICE)


