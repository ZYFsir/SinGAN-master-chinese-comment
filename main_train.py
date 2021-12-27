from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)         # 根据不同任务，生成不同的结果存储路径

    if (os.path.exists(dir2save))==False:   # TODO:此处修改了一下便于debug，正式运行时记得删除==False
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real = functions.read_image(opt)                # 获取原图像
        functions.adjust_scales2image(real, opt)        # 保证输入图像尺寸在一定范围内
        train(opt, Gs, Zs, reals, NoiseAmp)             # 模型训练
        SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt)       # 生成不同样本
