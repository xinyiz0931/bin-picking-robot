import os
import time
import glob
import matplotlib.pyplot as plt
import termios
import numpy as np
from scipy import interpolate
from scipy.signal import find_peaks
from bpbot.config import BinConfig
import warnings
warnings.filterwarnings("ignore")

class FTSensor(object):
    def __init__(self):
        self.port = (glob.glob(r'/dev/ttyUSB*'))[0]
        self.force_sensitivity = 32.765 # unit: LSB/N
        self.torque_sensitivity = 1638.250 # unit: LSB/Nm
        self.tw = 20 # unit: ms
        # self.tw = 100 # unit: ms
        self.force_lm = 10 # unit: N
        self.torque_lm = 0.5 # unit: Nm
        
        cfg = BinConfig()
        zero_path = os.path.join(cfg.root_dir, "data/force/zero.txt")
        self.zero_load = np.mean(np.loadtxt(zero_path), axis=0)[1:]
        self.out_path = os.path.join(cfg.root_dir, "data/force/out.txt")
        self.plot_path = os.path.join(cfg.root_dir, "data/force/out.png")
        self.colors = [
            [255.0/255.0, 87.0/255.0, 89.0/255.0], # 0_red
            [ 89.0/255.0,169.0/255.0, 79.0/255.0], # 1_green
            [ 78.0/255.0,121.0/255.0,167.0/255.0], # 2_blue
            [237.0/255.0,201.0/255.0, 72.0/255.0], # 3_yellow
            [242.0/255.0,142.0/255.0, 43.0/255.0], # 4_orange
            [176.0/255.0,122.0/255.0,161.0/255.0], # 5_purple
            [255.0/255.0,157.0/255.0,167.0/255.0], # 6_pink
            [118.0/255.0,183.0/255.0,178.0/255.0], # 7_cyan
            [156.0/255.0,117.0/255.0, 95.0/255.0], # 8_brown
            [186.0/255.0,176.0/255.0,172.0/255.0]] # 9_gray

        self.fig = plt.figure(1, figsize=(16, 10))
        self.sep_kw =[-1]*7
    
    def connect(self):
        try:
            fdc = os.open(self.port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
            print("Open port "+self.port)
        except BlockingIOError as e:
            print("Can't open port!", e)
            fdc = -1

        if (fdc < 0):
            os.close(fdc)

        # tty control
        term_ = termios.tcgetattr(fdc)
        term_[0] = termios.IGNPAR #iflag
        term_[1] = 0 # oflag
        term_[2] = termios.B921600 | termios.CS8 | termios.CLOCAL | termios.CREAD # cflag
        term_[3] = 0 # lflag -> ICANON
        term_[4] = 4103 # ispeed
        term_[5] = 4103 # ospeed
        # # cc
        o_ = bytes([0])
        term_[6][termios.VINTR] = o_ # Ctrl-c
        term_[6][termios.VQUIT] = o_ # Ctrl-?
        term_[6][termios.VERASE] = o_ # del
        term_[6][termios.VKILL] = o_ # @
        term_[6][termios.VEOF] =  bytes([4])# Ctrl-d
        term_[6][termios.VTIME] = 0
        term_[6][termios.VMIN] = 0
        term_[6][termios.VSWTC] = o_ # ?0
        term_[6][termios.VSTART] = o_ # Ctrl-q
        term_[6][termios.VSTOP] = o_ # Ctrl-s
        term_[6][termios.VSUSP] = o_ # Ctrl-z
        term_[6][termios.VEOF] = o_ # ?0
        term_[6][termios.VREPRINT] = o_ # Ctrl-r
        term_[6][termios.VDISCARD] = o_ # Ctrl-u
        term_[6][termios.VWERASE] = o_ # Ctrl-w
        term_[6][termios.VLNEXT] = o_ # Ctrl-v
        term_[6][termios.VEOL2] = o_ # ?0
        termios.tcsetattr(fdc, termios.TCSANOW, term_)
        return fdc

    def smooth(self, x, y):
        k = int(len(x))

        # x_new = np.linspace(x.min(), x.max(), k)  
        x_new = x
        y_new = []
        if len(y.shape) == 1:
            tck = interpolate.splrep(x, y, s=0.1)
            y_new = interpolate.splev(x_new, tck, der=0)
        else:
            for i in range(y.shape[1]):
                # smooth = interpolate.interp1d(x, y[:,i], kind='cubic')
                # smooth = UnivariateSpline(x, y[:,i])
                # smooth.set_smoothing_factor(0.2)
                # y_new.append(smooth(x_new))

                tck = interpolate.splrep(x, y[:,i], s=0.1)
                y_new.append(interpolate.splev(x_new, tck, der=0))
            y_new = np.asarray(y_new).T
        return x_new, y_new
    
    def filter(self, x, y):
        # k=21
        # k = len(x) if len(x)%2!=0 else len(x) -1
        k = 39
        from scipy.signal import savgol_filter
        x_new = x
        y_new = []
        if len(y.shape) == 1:
            y_new = np.asarray(savgol_filter(y, k, 2)).T
        else:
            for i in range(y.shape[1]):
                y_new.append(savgol_filter(y[:,i], k, 2))
            y_new = np.asarray(y_new).T
        return x_new, y_new

    def examine(self):
        def func(x, a, b, c):
            return a * np.exp(-b * x) + c
        _data = np.loadtxt("/home/hlab/bpbot/data/force/all_success_human_grasp/3/outa.txt")
        tms = _data[:,0]
        ft = _data[:,1:]
        ft[:,:3] = (ft[:,:3]-self.zero_load[:3])/self.force_sensitivity
        ft[:,3:] = (ft[:,3:]-self.zero_load[3:])/self.torque_sensitivity
        time_start = tms[0] - 100 # [ms]
        tms -= time_start
        T = 500 # [ms]
        j=0
        for i, t in enumerate(tms):
            print("------------", i)
            if t % T == 0:
                Fz = ft[j:(i+1), 2]
                tm = tms[j:(i+1)]
                j=i
                from scipy.optimize import curve_fit
                from scipy.stats import linregress

                fit = linregress(tm, Fz)
                slope, intercept, r_value, p_value, std_err = linregress(tm, Fz)
                print("slope: %f    mean: %f"% (slope, np.mean(Fz)))
                # print("R-squared: %f" % r_value**2)
                plt.plot(tm, Fz, 'o', label='original data')
                plt.plot(tm, intercept + slope*tm, 'r', label='fitted line')
                plt.legend()
                
                # tm_, Fz_ = self.smooth(tm, Fz)
                # print(tm.shape, Fz.shape, "->", tm_.shape, Fz_.shape)
                plt.plot(tm, Fz, 'b-', alpha=0.5)
                # plt.plot(tm_, Fz_, 'b-')
                plt.ylim(-self.force_lm, self.force_lm)
                
                plt.show()


    def plot_interval(self, filepath=None):
        if filepath is not None:
            _data = np.loadtxt(filepath)
        else:
            _data = np.loadtxt(self.out_path)
        
        if not self.sep_kw in _data:
            print("Data is not available! ")
            return
        sep_x = []
        sep_idx = np.where(np.any(_data==self.sep_kw, axis=1))[0]
        i = 1
        s_i = sep_idx[-(i+1)]
        e_i = sep_idx[-i]
        data = _data[s_i+1:e_i]

        tm = data[:,0]/1000
        ft = data[:,1:]
        ft[:,:3] = (ft[:,:3]-self.zero_load[:3])/self.force_sensitivity
        ft[:,3:] = (ft[:,3:]-self.zero_load[3:])/self.torque_sensitivity

        # ft = (data[:,1:]-self.zero_load)/1000
        x = tm - tm[0]

        # tm_, ft_ = self.smooth(tm, ft)
        tm_, ft_ = self.filter(tm, ft)
        x_ = tm_ - tm_[0]

        fig = plt.figure(1, figsize=(6, 6))

        ax1 = fig.add_subplot(211)
        ax1.set_prop_cycle(color=self.colors[:3])
        # major_ticks = np.arange(x[0], x[-1], 10)
        # ax1.set_xticks(major_ticks)
        ax1.plot(x, ft[:,:3], alpha=0.3)
        ax1.plot(x_, ft_[:,:3])

        ax1.grid(which='minor', linestyle='dotted', alpha=.5)
        ax1.grid(which='major', linestyle='dotted', alpha=1)
        ax1.legend(['Fx', 'Fy', 'Fz'], loc='upper right')
        ax1.set_title('Force')
        plt.ylim(-self.force_lm, self.force_lm)
        
        ax2 = fig.add_subplot(212)
        ax2.set_prop_cycle(color=self.colors[:3])
        # major_ticks = np.arange(x[0], x[-1], 10)
        # ax2.set_xticks(major_ticks)
        ax2.plot(x, ft[:,3:], alpha=0.3)
        ax2.plot(x_, ft_[:,3:])

        # # find peaks
        # peaks_mx, _ = find_peaks(ft_[:,3], height=0.05)
        # peaks_my, _ = find_peaks(ft_[:,4], height=0.05)
        # peaks_mz, _ = find_peaks(ft_[:,5], height=0.05)
        # ax2.plot(x_[peaks_mx], ft_[:,3][peaks_mx], "x", color='r')
        # ax2.plot(x_[peaks_my], ft_[:,4][peaks_my], "x", color='g')
        # ax2.plot(x_[peaks_mz], ft_[:,5][peaks_mz], "x", color='b')
        # #find valleys
        # valleys_mx, _ = find_peaks(-1*ft_[:,3], height=-0.05)
        # valleys_my, _ = find_peaks(-1*ft_[:,4], height=-0.05)
        # valleys_mz, _ = find_peaks(-1*ft_[:,5], height=-0.05)
        # ax2.plot(x_[valleys_mx], ft_[:,3][valleys_mx], "x", color='r')
        # ax2.plot(x_[valleys_my], ft_[:,4][valleys_my], "x", color='g')
        # ax2.plot(x_[valleys_mz], ft_[:,5][valleys_mz], "x", color='b')

        # ax1.axhline(y=0.25, color=colors[7], alpha=.7, linestyle='dashed')
        ax2.grid(which='minor', linestyle='dotted', alpha=.5)
        ax2.grid(which='major', linestyle='dotted', alpha=1)
        
        for sx in sep_x:
            ax1.axvline(x=sx, color=self.colors[3], alpha=.7, linestyle='dashed')
            ax2.axvline(x=sx, color=self.colors[3], alpha=.7, linestyle='dashed')
        ax2.legend(['Mx', 'My', 'Mz'], loc='upper right')
        ax2.set_title('Torque')
        plt.ylim(-self.torque_lm, self.torque_lm)
        plt.savefig(self.plot_path)
        print("Force: ", ft[-1][:3])
        print("Torque: ", ft[-1][3:])
        print("Save plotted result!")
        # plt.show()
        return ft[-1][2], ft[-1][3], ft[-1][4]

    def plot_file(self, _data=None, _path=None):
        if _path is None and _data is None:
            _path = self.out_path

        if _data is None:
            _data = np.loadtxt(_path)
            
        sep_x = []
        if self.sep_kw in _data:
            sep_idx = np.where(np.any(_data==self.sep_kw, axis=1))[0]
            data = np.delete(_data, sep_idx, 0)
            sep_idx -= 1
            sep_x = _data[sep_idx][:,0]/1000
        else:
            data = _data

        tm = data[:,0]/1000
        ft = data[:,1:]
        ft[:,:3] = (ft[:,:3]-self.zero_load[:3])/self.force_sensitivity
        ft[:,3:] = (ft[:,3:]-self.zero_load[3:])/self.torque_sensitivity

        # ft = (data[:,1:]-self.zero_load)/1000
        x = tm - tm[0]

        # tm_, ft_ = self.smooth(tm, ft)
        # tm_, ft_ = self.filter(tm, ft)
        
        x_ = x
        ft_ = ft
        # x_ = tm_ - tm_[0]

        fig = plt.figure(1, figsize=(16, 10))

        ax1 = fig.add_subplot(211)
        ax1.set_prop_cycle(color=self.colors[:3])
        # major_ticks = np.arange(x[0], x[-1], 10)
        # ax1.set_xticks(major_ticks)
        ax1.plot(x, ft[:,:3], alpha=0.2)
        ax1.plot(x_, ft_[:,:3])

        ax1.grid(which='minor', linestyle='dotted', alpha=.5)
        ax1.grid(which='major', linestyle='dotted', alpha=1)
        ax1.legend(['Fx', 'Fy', 'Fz'], loc='upper right')
        ax1.set_title('Force')
        plt.ylim(-self.force_lm, self.force_lm)
        
        ax2 = fig.add_subplot(212)
        ax2.set_prop_cycle(color=self.colors[:3])
        # major_ticks = np.arange(x[0], x[-1], 10)
        # ax2.set_xticks(major_ticks)
        ax2.plot(x, ft[:,3:], alpha=0.2)
        ax2.plot(x_, ft_[:,3:])

        # # find peaks
        # peaks_mx, _ = find_peaks(ft_[:,3], height=0.05)
        # peaks_my, _ = find_peaks(ft_[:,4], height=0.05)
        # peaks_mz, _ = find_peaks(ft_[:,5], height=0.05)
        # ax2.plot(x_[peaks_mx], ft_[:,3][peaks_mx], "x", color='r')
        # ax2.plot(x_[peaks_my], ft_[:,4][peaks_my], "x", color='g')
        # ax2.plot(x_[peaks_mz], ft_[:,5][peaks_mz], "x", color='b')
        # #find valleys
        # valleys_mx, _ = find_peaks(-1*ft_[:,3], height=-0.05)
        # valleys_my, _ = find_peaks(-1*ft_[:,4], height=-0.05)
        # valleys_mz, _ = find_peaks(-1*ft_[:,5], height=-0.05)
        # ax2.plot(x_[valleys_mx], ft_[:,3][valleys_mx], "x", color='r')
        # ax2.plot(x_[valleys_my], ft_[:,4][valleys_my], "x", color='g')
        # ax2.plot(x_[valleys_mz], ft_[:,5][valleys_mz], "x", color='b')

        # ax1.axhline(y=0.25, color=colors[7], alpha=.7, linestyle='dashed')
        ax2.grid(which='minor', linestyle='dotted', alpha=.5)
        ax2.grid(which='major', linestyle='dotted', alpha=1)
        
        for sx in sep_x:
            ax1.axvline(x=sx, color=self.colors[3], alpha=.7, linestyle='dashed')
            ax2.axvline(x=sx, color=self.colors[3], alpha=.7, linestyle='dashed')
        ax2.legend(['Mx', 'My', 'Mz'], loc='upper right')
        ax2.set_title('Torque')
        plt.ylim(-self.torque_lm, self.torque_lm)
        plt.savefig(self.plot_path)
        print("Save plotted result!")
        # plt.show()

    def plot(self, ft, tm):
        # fig = plt.figure(1, figsize=(16, 10))
        ft = np.array(ft)
        ax1 = self.fig.add_subplot(211)
        ax1.set_prop_cycle(color=self.colors[:3])
        major_ticks = np.arange(tm[0], tm[-1], 10)
        
        tm_, ft_ = self.smooth(tm, ft)
        
        ax1.set_title('Force')
        ax1.set_xticks(major_ticks)
        # ax3.axhline(y=0.25, color=colors[7], alpha=.7, linestyle='dashed')

        ax1.grid(which='minor', linestyle='dotted', alpha=.5)
        ax1.grid(which='major', linestyle='dotted', alpha=1)
        ax1.plot(tm, ft[:,:3], alpha=0.4)
        ax1.plot(tm_, ft_[:,:3])
        
        ax1.legend(['Fx', 'Fy', 'Fz'], loc='upper right')
        plt.ylim(-self.force_lm, self.force_lm)
        
        ax2 = self.fig.add_subplot(212)
        ax2.set_prop_cycle(color=self.colors[:3])
        major_ticks = np.arange(tm[0], tm[-1], 10)
        
        ax2.set_title('Torque')
        ax2.set_xticks(major_ticks)

        ax2.grid(which='minor', linestyle='dotted', alpha=.5)
        ax2.grid(which='major', linestyle='dotted', alpha=1)
        ax2.plot(tm, ft[:,3:], alpha=0.4)
        ax2.plot(tm_, ft_[:,3:])

        ax2.legend(['Mx', 'My', 'Mz'], loc='upper right')
        plt.ylim(-self.torque_lm, self.torque_lm)
        plt.ion()
        plt.show()

    def record(self, plot=False, stop=0):
        fdc = self.connect()
        clkb = 0
        clk0 = (time.process_time())*1000 # ms
        data = []
        j = 0
        r_ = str.encode("R")
        os.write(fdc, r_)

        fz = []
        k=0

        while True:
            if j == 0:
                open(self.out_path, 'w').close()

            line = []
            while True:
                clk = (time.process_time()) * 1000 - clk0
                if clk >= (clkb + self.tw):
                    clkb = clk / self.tw * self.tw
                    break
            os.write(fdc, r_)
            l = os.read(fdc, 27)
            time_stamp = int(clk / self.tw * self.tw)
            if l == bytes(): continue

            line.append(time_stamp)

            for i in range(1,22,4):
                line.append(int((l[i:i+4]).decode(),16))
            detected_load = line[1:].copy()
            detected_load[:3] = (detected_load[:3]-self.zero_load[:3])/self.force_sensitivity
            detected_load[3:] = (detected_load[3:]-self.zero_load[3:])/self.torque_sensitivity
            with open(self.out_path, 'a') as fp:
                print(*line, file=fp)

            data.append(detected_load)
            
            j+=1 
            if plot:
                
                if len(data) <= 50:
                    ft = [[0,0,0,0,0,0] for _ in range(50-j)] + data
                else:
                    ft = data[j-50:]

                plt.clf()
                # self.plot(ft, range(j,j+50))
                self.plot(ft, np.arange(j,j+50))
                plt.pause(0.005)
            # else:
            #     print(line)
            # T = 200
            # fz_before = -9999
            # if time_stamp >= 5380 and time_stamp <= 8380:
            #     fz.append(detected_load[2])
            #     if time_stamp % T ==0:
            #         fz = np.mean(fz[k:T*(k+1)])
            #         k+=1
            #         if fz < fz_before or (fz > fz_before and fz > 0.8):
            #             os.system("bash /home/hlab/bpbot/script/stop_ft.sh")
            #             print("BREAK!")
            #             break

            if stop > 0 and line[0] > stop*1000:
                break
        



if __name__ == "__main__":
    sensor = FTSensor()
    # sensor.examine()
    sensor.record(plot=True)
    # sensor.plot_file()

