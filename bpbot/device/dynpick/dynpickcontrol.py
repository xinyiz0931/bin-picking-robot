import os
import shutil
import time
import glob
import matplotlib.pyplot as plt
import termios
import numpy as np
from bpbot.config import BinConfig
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
# from scipy.signal import find_peaks
import warnings
warnings.filterwarnings("ignore")

class DynPickControl(object):
    def __init__(self):
        self.port = (glob.glob(r'/dev/ttyUSB*'))[0]
        self.force_sensitivity = 32.765 # unit: LSB/N
        self.torque_sensitivity = 1638.250 # unit: LSB/Nm
        self.tw = 50 # unit: ms
        self.force_lm = 7.5 # unit: N
        self.torque_lm = 0.5 # unit: Nm
        
        cfg = BinConfig()
        self.zero_path = os.path.join(cfg.root_dir, "data/force/zero.txt")
        self.zero_load = np.mean(np.loadtxt(self.zero_path), axis=0)[1:]
        self.out_path = os.path.join(cfg.root_dir, "data/force/out.txt")
        self.plot_path = os.path.join(cfg.root_dir, "data/force/vis.png")

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

        self.sep_kw =[-1]*7

    def connect(self):
        try:
            fdc = os.open(self.port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
            print("Open port "+self.port)
            print("Frequency:", 1/(self.tw/1000), "Hz")
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
    
    def fit(self, y, vis=False):
        # def func(x,a,b,c):
        #     return a/(b+np.exp(-b*x))+c
        # def func(x,a,b,c,d):
        #     return a/(d+np.exp(-b*x))+c 
        def nonlinear(x,a,b,c,d):
            # return d+(a-d)/((1+(x/c)**b))**m
            return d+(a-d)/(1+(x/c)**b)

        def linear(x,a,b):
            return a*x + b
        y = np.array(y)
        # replace all negative numbers
        y[y<0]=0
        # x = np.arange(y.shape[0])
        x = np.linspace(0, 0.4, y.shape[0])
        # _, y = self.filter(y,x, method="median", param=11)
        
        
        popt, pcov = curve_fit(linear, x, y)
        y_ = linear(x, *popt)
        if np.abs(popt[0]) < 0.004 and np.abs(popt[1]) < 0.3:
        # if popt[0] < 0.01 and np.abs(popt[1]) < 0.1:
            flag = True
        else:
            flag = False
        # else:
        #     print("Linear fitting failed! ")
        #     popt, pcov = curve_fit(nonlinear, x, y)
        #     y_ = nonlinear(x, *popt)
        #     flag = False

        # try:
        #     popt, pcov = curve_fit(nonlinear, x, y)
        #     y_ = nonlinear(x, *popt)
        # except:
        #     print("Fit linear")
        #     popt, pcov = curve_fit(linear, x, y)
        #     y_ = linear(x, *popt)

        # fig = plt.figure(figsize=(9,3))
        if vis:
            plt.axhline(y=1.5, color='gold', alpha=.7, linestyle='dashed')
            plt.plot(x, y, alpha=0.4)
            plt.plot(x, y_)
            plt.ylim([-10,10])
            plt.title(np.round(popt,3))
            plt.show()
        # return flag
        return popt

        # try: 
        #     popt, pcov = curve_fit(nonlinear, x, y)
        #     y_ = nonlinear(x, *popt)
        # except Exception:
        #     print("Fit linear")
        #     popt, pcov = curve_fit(linear, x, y)
        #     y_ = linear(x, *popt)
        # return popt
    
    def calib(self):
        # record for 5 seconds
        self.record(plot=False, stop=5) 
        # reset zero point
        shutil.copyfile(self.out_path, self.zero_path)
        print("Reset zero point! ")

    def smooth(self, x, y):
        k = int(len(x))
        if isinstance(y, list): y = np.array(y)
        # x_new = np.linspace(x.min(), x.max(), k)  
        x_new = np.linspace(x.min(), x.max(), int(k/2))
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

                tck = interpolate.splrep(x, y[:,i], s=0.6)
                y_new.append(interpolate.splev(x_new, tck, der=0))
            y_new = np.asarray(y_new).T
        return x_new, y_new
    
    def filter(self, y, x=None, method="median", param=11):
        # k=21
        # k = len(x) if len(x)%2!=0 else len(x) -1
        # k = 39
        # x = np.arange(len(y))
        if x is None:
            x = np.arange(len(y))
        if len(x) <= 11:
            param = len(x) if len(x)%2 != 0 else len(x)-1
        if isinstance(y, list):
            y = np.array(y)
        x_new = x
        y_new = []
        if len(y.shape) == 1:
            if method == "savgol":
                y_new = np.asarray(savgol_filter(y, param, 2)).T
            elif method == "median":
                y_new = np.asarray(self.median_filter_1d(y, param)).T
        else:
            if method == "savgol":
                for i in range(y.shape[1]):
                    y_new.append(savgol_filter(y[:,i], param, 2))
            elif method == "median":
                for i in range(y.shape[1]):
                    y_new.append(self.median_filter_1d(y[:,i], param)) 
            y_new = np.asarray(y_new).T
        return x_new, y_new

    def median_filter_1d(self, x, k):
        x = np.array(x)
        # assert k % 2 == 1, "Median filter length must be odd."
        # assert x.ndim == 1, "Input must be one-dimensional."
        k2 = (k - 1) // 2
        y = np.zeros ((len (x), k), dtype=x.dtype)
        y[:,k2] = x
        for i in range (k2):
            j = k2 - i
            y[j:,i] = x[:-j]
            y[:j,i] = x[0]
            y[:-j,-(i+1)] = x[j:]
            y[-j:,-(i+1)] = x[-1]
        return np.median(y, axis=1)

    def plot(self, _data=None, _path=None, filter=False, animation=False):
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
        
        
        if _data[0][0] == 0: 
            data = _data[1:,]
            print("Data contains reset f0!")
            f0 = _data[0,1:]
        else: 
            f0 = self.zero_load
        if data.shape[1] == 7:
            tm = data[:,0]/1000
            ft = data[:,1:]
            ft[:,:3] = (ft[:,:3]-f0[:3])/self.force_sensitivity
            ft[:,3:] = (ft[:,3:]-f0[3:])/self.torque_sensitivity
        elif data.shape[1] == 6:
            tm = np.arange(data.shape[0])
            ft = data
        
        # ft = (data[:,1:]-self.zero_load)/1000
        x = tm - tm[0]

        # tm_, ft_ = self.smooth(tm, ft)
        if filter: 
            tm_, ft_ = self.filter(ft, tm, method="median", param=11)
            x_ = tm_ - tm_[0]

        fig = plt.figure(1, figsize=(16, 10))

        ax1 = fig.add_subplot(211)
        ax1.set_prop_cycle(color=self.colors[:3])
        # major_ticks = np.arange(x[0], x[-1], 10)
        # ax1.set_xticks(major_ticks)
        if filter:
            ax1.plot(x, ft[:,:3], alpha=0.2)
            ax1.plot(x_, ft_[:,:3])
        else:
            ax1.plot(x, ft[:,:3])

        ax1.grid(which='minor', linestyle='dotted', alpha=.5)
        ax1.grid(which='major', linestyle='dotted', alpha=1)
        ax1.legend(['Fx', 'Fy', 'Fz'], loc='upper right')
        ax1.set_title('Force')
        ax1.set_xticks(np.arange(tm[0], tm[-1], 10))
        ax1.set_yticks(np.arange(-self.force_lm, self.force_lm, 0.5))
        ax1.axhline(y=0, color=self.colors[-1], alpha=.7, linestyle='dashed')
        ax1.axhline(y=2, color='gold', alpha=.7, linestyle='dashed')
        
        ax2 = fig.add_subplot(212)
        ax2.set_prop_cycle(color=self.colors[:3])
        # major_ticks = np.arange(x[0], x[-1], 10)
        # ax2.set_xticks(major_ticks)
        if filter:
            ax2.plot(x, ft[:,3:], alpha=0.2)
            ax2.plot(x_, ft_[:,3:])
        else:
            ax2.plot(x, ft[:,3:])

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
        ax2.set_xticks(np.arange(tm[0], tm[-1], 10))
        ax2.set_yticks(np.arange(-self.torque_lm, self.torque_lm, 0.1))
        ax2.axhline(y=0, color=self.colors[-1], alpha=.7, linestyle='dashed')
        # plt.savefig(self.plot_path)
        # print("Save plotted result!")
        plt.show()
        j = 0
        fig2 = plt.figure(1, figsize=(8, 8))
        itvl = 20
        if animation:
            data = list(ft_)
            # data = [ft[:,2]]
            while j < len(data)-1:
                if len(data) <= itvl:
                    ft = [[0,0,0,0,0,0] for _ in range(itvl-j)] + data
                else:
                    ft = data[j:j+itvl]
                plt.clf()
                self.plot_data(ft, fig=fig2)
                plt.pause(0.005)
                j+=1 
                
            # else:
            #     print(line)

    def plot_data(self, ft, tm=None, fig=None):
        # fig = plt.figure(1, figsize=(16, 10))
        ft = np.array(ft)
        if tm is None:
            tm = np.arange(ft.shape[0])
        if fig is None:
            fig = self.fig

        ax1 = fig.add_subplot(211)
        ax1.set_prop_cycle(color=self.colors[:3])
        
        # tm_, ft_ = self.smooth(tm, ft)
        tm_, ft_ = self.filter(ft, tm, method="median", param=5)
        # ft_ = self.median_filter_1d(ft)
        
        ax1.set_title('Force')
        ax1.set_xticks(np.arange(tm[0], tm[-1], 10))
        ax1.set_yticks(np.arange(-self.force_lm, self.force_lm, 1))
        # ax1.axhline(y=0, color='grey', alpha=.7, linestyle='dashed')


        ax1.grid(which='minor', linestyle='dotted', alpha=.5)
        ax1.grid(which='major', linestyle='dotted', alpha=1)
        ax1.plot(tm, ft[:,:3], alpha=0.4)
        ax1.plot(tm_, ft_[:,:3])
        # ax1.plot(tm_, ft_[:,2], color=self.colors[2])
        
        ax1.legend(['Fz', 'Fy', 'Fz'], loc='upper right')
        plt.ylim(-self.force_lm, self.force_lm)
        
        ax2 = fig.add_subplot(212)
        ax2.set_prop_cycle(color=self.colors[:3])
        major_ticks = np.arange(tm[0], tm[-1], 10)
        
        ax2.set_title('Torque')
        ax2.set_xticks(np.arange(tm[0], tm[-1], 10))
        ax2.set_yticks(np.arange(-self.torque_lm, self.torque_lm, 0.1))
        ax2.axhline(y=0, color=self.colors[-1], alpha=.7, linestyle='dashed')
        

        ax2.grid(which='minor', linestyle='dotted', alpha=.5)
        ax2.grid(which='major', linestyle='dotted', alpha=1)
        ax2.plot(tm, ft[:,3:], alpha=0.4)
        ax2.plot(tm_, ft_[:,3:])

        ax2.legend(['Mx', 'My', 'Mz'], loc='upper right')
        ax1.axhline(y=2, color="gold", alpha=.7, linestyle='dashed')
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
        self.fig = plt.figure(1, figsize=(16, 10))

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
            
            print(line, end='\r')

            data.append(detected_load)
            
            j+=1 
            if plot:
                if len(data) <= 50:
                    ft = [[0,0,0,0,0,0] for _ in range(50-j)] + data
                else:
                    ft = data[j-50:]

                plt.clf()
                self.plot_data(ft, np.arange(j,j+50))
                plt.pause(0.005)
            # else:
            #     print(line)

            if stop > 0 and line[0] > stop*1000:
                break

def main():

    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--r','-r', type=str, help='action you want', default="vis")
    args = parser.parse_args()

    sensor = DynPickControl()
    
    if args.r == "monitoring":
        sensor.record(plot=False)
    elif args.r == "calib":
        sensor.calib()
    elif os.path.exists(args.r):
        sensor.plot(_path=args.r, filter=True)
    elif args.r == "vis":
        sensor.record(plot=True)
    else:
        print("Wrong requirement for dynpick...")

if __name__ == "__main__":
    main()
