import torch
import torch.nn as nn
import torch.nn.functional as F
from dask.config import set

from .backbone import resnet18
from .fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion
from .res3d import r3d_18, r2plus1d_18, mc3_18
from .resnet1d import resnet18_1d




class AVClassifier(nn.Module):
    def __init__(self, args):
        super(AVClassifier, self).__init__()

        self.args = args
        fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        elif args.dataset == '360x':
            n_classes = 21
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))
        self.B = args.batch_size
        self.n_classes = n_classes

        self.use_stereo_audio = args.use_stereo_audio
        
        self.model_n = 0
        self.backbone_embed = 512

        self.fc_embed = 128
        self.au_embed = 0

        self.use_front = args.use_front
        self.use_360 = args.use_360
        self.use_clip = args.use_clip
        self.use_audio = args.use_audio
        self.use_shared_audio = args.use_shared_audio
        self.use_at = args.use_directional_audio

        self.dropout_rate = 0.1
        
        modality = 1

        if self.use_audio:
            self.au_embed = 64
        
            # if self.use_stereo_audio:
                
        if self.use_at:
            self.at_net = self.get_at_net()

        if args.use_360:
            self.x360_net = self.get_backbone()

        if args.use_clip:
            self.clip_net = self.get_backbone()

        if args.use_front:
            self.front_net = self.get_backbone()

        if self.use_shared_audio:

            self.audio_net = resnet18(modality='audio')
            self.audio_fc = nn.Sequential(

                nn.ReLU(),  # inplace=True
                nn.Linear(self.backbone_embed, self.fc_embed),
                nn.LayerNorm(self.fc_embed),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate)

                # nn.Linear(256, self.n_classes)  # 16 * 2
            )

            if self.args.aux_loss:
                self.audio_head = nn.Linear(self.fc_embed, self.n_classes)


        au_embed = self.au_embed
        if self.use_stereo_audio:
            au_embed += self.au_embed

        fused_embed = (au_embed + self.fc_embed) * self.model_n

        if self.use_at:
            fused_embed += self.au_embed

        print("fused_embed :", fused_embed)  # 960


        # modulation
        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':


            self.fusion_module = ConcatFusion(
                                   input_dim=fused_embed,
                                   output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':  # This is the default fusion method
            self.fusion_module = GatedFusion(input_dim=self.model_n * self.fc_embed ,
                                             output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

    
    def forward_at_net(self, at):
        out_dict = {}
        at = self.at_net['at'](at)
        at = F.adaptive_avg_pool1d(at, 1)
        at = torch.flatten(at, 1)

        # print("at out feat:", at.shape)   # [5, 512]

        at_feat = self.at_net['at_fc'](at)
        out_dict['at_feat'] = at_feat
        
        at_out = self.at_net['at_head'](at_feat)
        out_dict['at_out'] = at_out
        
        return out_dict
            
    
    def get_at_net(self):
        at_net = {}
        at_net['at'] = resnet18_1d(modality='audio')
        at_net['at_fc'] = nn.Sequential(
                # nn.Dropout(self.dropout_rate), 
                nn.ReLU(),  # inplace=True
                nn.Linear(self.backbone_embed, self.au_embed),
                nn.Dropout(self.dropout_rate),
                nn.LayerNorm(self.au_embed),
                nn.ReLU(inplace=True)
                # nn.Dropout(self.dropout_rate)
                # nn.Linear(256, self.n_classes)  # 16 * 2
            )
        
        at_net['at_head'] = nn.Linear(self.au_embed, self.n_classes)
                
        return nn.ModuleDict(at_net)
    
    def get_backbone(self):
        backbone = {}
        visual_net = r3d_18(pretrained=False, progress=True)
        # resnet18(modality='visual')

        self.model_n += 1
        fc = self.get_fc()
        if self.use_audio and not self.use_shared_audio:
            audio_net = resnet18(modality='audio')
            backbone['audio'] = audio_net
            if self.use_stereo_audio:
                backbone['audio_stereo'] = resnet18(modality='audio')

        backbone['visual'] = visual_net
        backbone['fc'] = fc
        return nn.ModuleDict(backbone)

    def get_fc(self):
        fc = {}
        # Conv - DropOut - BatchNorm - Activation - Pool

        visual_fc = nn.Sequential(
                        # nn.Dropout(self.dropout_rate), 
                        nn.ReLU(),  # inplace=True
                        nn.Linear(self.backbone_embed, self.fc_embed),
                        nn.Dropout(self.dropout_rate),
                        nn.LayerNorm(self.fc_embed),
                        nn.ReLU(inplace=True)
                        
                    )
        fc['visual'] = visual_fc

        if self.args.aux_loss:
            visual_head = nn.Linear(self.fc_embed, self.n_classes)
            fc['visual_head'] = visual_head

        if self.use_audio and not self.use_shared_audio:
            audio_fc = nn.Sequential(
                # nn.Dropout(self.dropout_rate), 
                nn.ReLU(),  # inplace=True
                nn.Linear(self.backbone_embed, self.au_embed),
                nn.Dropout(self.dropout_rate),
                nn.LayerNorm(self.au_embed),
                nn.ReLU(inplace=True)
                # nn.Dropout(self.dropout_rate)
                # nn.Linear(256, self.n_classes)  # 16 * 2
            )
            fc['audio'] = audio_fc
            if self.args.aux_loss:
                audio_head = nn.Linear(self.au_embed, self.n_classes)
                fc['audio_head'] = audio_head
            
            if self.use_stereo_audio:
                audio_fc_stereo = nn.Sequential(
                    # nn.Dropout(self.dropout_rate), 
                    nn.ReLU(),  # inplace=True
                    nn.Linear(self.backbone_embed, self.au_embed),
                    nn.Dropout(self.dropout_rate),
                    nn.LayerNorm(self.au_embed),
                    nn.ReLU(inplace=True)
                    # nn.Dropout(self.dropout_rate)
                    # nn.Linear(256, self.n_classes)  # 16 * 2
                )
                fc['audio_stereo'] = audio_fc_stereo
                if self.args.aux_loss:
                    audio_head_stereo = nn.Linear(self.au_embed, self.n_classes)
                    fc['audio_head_stereo'] = audio_head_stereo
            

        return nn.ModuleDict(fc)

    def forward_branch(self, audio, visual, net, at=None):
        
        out_dict = {}
        
        if self.use_audio:
            if self.use_shared_audio:
                a = self.audio_net(audio[0])
            else:
                a = net['audio_stereo'](audio[0])
            a = F.adaptive_avg_pool2d(a, 1)
            a = torch.flatten(a, 1)

            if self.use_stereo_audio:
                if self.use_shared_audio:
                    a2 = self.audio_net(audio[1])
                else:
                    a2 = net['audio_stereo'](audio[1])
                a2 = F.adaptive_avg_pool2d(a2, 1)
                a2 = torch.flatten(a2, 1)

        # Visual
        v = net['visual'](visual)

        (_, C, T, H, W) = v.size()
        B = v.size()[0] #// 3
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(v, 1)
        v = torch.flatten(v, 1)

        if self.use_audio:
            if not self.use_shared_audio:
                a_feat = net['fc']['audio'](a)
            else:
                a_feat = self.audio_fc(a)

            if self.use_stereo_audio:
                if not self.use_shared_audio:
                    a_feat2 = net['fc']['audio_stereo'](a2)
                else:
                    a_feat2 = self.audio_fc(a2)
                out_dict['a_feat2'] = a_feat2

            out_dict['a_feat'] = a_feat

            if self.args.aux_loss:
                if not self.use_shared_audio:
                    a_out = net['fc']['audio_head'](a_feat)
                else:
                    a_out = self.audio_head(a_feat)

                if self.use_stereo_audio:
                    if not self.use_shared_audio:
                            a_out2 = net['fc']['audio_head_stereo'](a_feat2)
                    else:
                        a_out2 = self.audio_head(a_feat2)
                
                    out_dict['audio_out_stereo'] = a_out2
                                  
                # a_out = fc['audio_head'](a_feat)
                out_dict['audio_out'] = a_out

        v_feat = net['fc']['visual'](v)
        out_dict['v_feat'] = v_feat

        if self.args.aux_loss:
            v_out = net['fc']['visual_head'](v_feat)
            out_dict['visual_out'] = v_out

        return out_dict

    def concat(self, f, x):
        if f == None:
            return x
        else:
            return torch.cat((f, x), dim=1)

    def store_output(self, cur_dict, label, a, v, out_dict):

        if self.use_audio:
            a = self.concat(a, cur_dict['a_feat'])
            if self.args.aux_loss:
                out_dict["a_" + label] = cur_dict['audio_out']

            if self.use_stereo_audio:
                a = self.concat(a, cur_dict['a_feat2'])
                if self.args.aux_loss:
                    out_dict["a2_" + label] = cur_dict['audio_out_stereo']
                
        v = self.concat(v, cur_dict['v_feat'])
        if self.args.aux_loss:
            out_dict["v_" + label] = cur_dict['visual_out']

        return a, v, out_dict

    def get_input(self, input_dict, device, label):

        a1 = input_dict[label]["a"][0].to(device).unsqueeze(1).float()
        a2 = input_dict[label]["a"][1].to(device).unsqueeze(1).float()
        audio = [a1, a2]
        
        visual = input_dict[label]["f"].to(device).float()

        return audio, visual

    def forward(self, input_dict, device):

        out_dict = {"a": [], "v": [], "out": []}
        a, v = None, None

        if self.use_clip:
            audio, visual = self.get_input(input_dict, device, "clip")

            cur_dict = self.forward_branch(audio, visual, self.clip_net)

            a, v, out_dict = self.store_output(cur_dict, "clip", a, v, out_dict)

        if self.use_front:
            audio, visual = self.get_input(input_dict, device, "front")

            cur_dict = self.forward_branch(audio, visual, self.front_net)

            a, v, out_dict = self.store_output(cur_dict, "front", a, v, out_dict)

        if self.use_360:
            audio, visual = self.get_input(input_dict, device, "360")

            cur_dict = self.forward_branch(audio, visual, self.x360_net)

            a, v, out_dict = self.store_output(cur_dict, "360", a, v, out_dict)

        if self.use_at:
            at_dict = self.forward_at_net(input_dict['at'])
            at_feat = at_dict['at_feat']
            a = self.concat(a, at_feat)
            out_dict["at_out"] = at_dict["at_out"]
            
        else:
            at_feat = None
            
        out_dict["fused"] = self.fusion_module( v, a, use_audio = self.args.use_audio)

        return out_dict


