import math
import numpy as np
import scipy
import scipy.signal
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from common import rcosdesign, save_fig, signal_freqz

def get_angle_plot(p1, p2, offset=1, color='b', len_x_axis=1, len_y_axis=1, label=""):
    angle1 = math.degrees(math.atan2(p1.imag, p1.real))
    angle2 = math.degrees(math.atan2(p2.imag, p2.real))

    theta1 = min(angle1, angle2)
    theta2 = max(angle1, angle2)
    return Arc([0, 0], len_x_axis*offset, len_y_axis*offset, 0, theta1, theta2,
               color=color, label=label)

def sync_error_plot():
    # plot tx signal
    plt.clf()
    signal = np.exp(1j*(2*np.pi/4*np.arange(4)+np.pi/4))
    plt.scatter(signal.real, signal.imag)
    plt.grid('on', ls=':')
    plt.xlabel('real')
    plt.ylabel('imag')
    save_fig(plt, '../doc/image/sync_rx_ideal.svg')

    plt.clf()
    signal = signal * np.exp(1j*np.pi/8)
    plt.scatter(signal.real, signal.imag)
    plt.grid('on', ls=':')
    plt.xlabel('real')
    plt.ylabel('imag')
    save_fig(plt, '../doc/image/sync_rx_phi.svg')

    plt.clf()
    symbol = np.floor(4*np.random.rand(100))
    signal = np.exp(1j*(2*np.pi/4*symbol+np.pi/4))
    signal = signal * np.exp(1j*np.pi*np.arange(100)/1000)
    plt.scatter(signal.real, signal.imag)
    plt.grid('on', ls=':')
    plt.xlabel('real')
    plt.ylabel('imag')
    save_fig(plt, '../doc/image/sync_rx_df.svg')

    plt.clf()
    signal = np.exp(1j*(np.pi/4))
    plt.scatter(signal.real, signal.imag)
    plt.gca().annotate(r'', xy=(0, 0), xytext=(signal.real, signal.imag),
                       arrowprops=dict(arrowstyle='<-'))
    plt.grid('on', ls=':')
    plt.xlabel('real')
    plt.ylabel('imag')
    rx = signal * np.exp(1j*np.pi/8)
    plt.gca().annotate(r'', xy=(0, 0), xytext=(rx.real, rx.imag),
                       arrowprops=dict(arrowstyle='<-'))
    plt.scatter(rx.real, rx.imag)
    plt.grid('on', ls=':')
    plt.xlabel('real')
    plt.ylabel('imag')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(['estimated signal', 'received signal'])
    angle_plot = get_angle_plot(signal, rx, 1)
    plt.gca().add_patch(angle_plot)
    plt.gca().text(0.3, 0.45, r'$\phi_e$')

    save_fig(plt, '../doc/image/sync_rx_phase_error.svg')

    angle_plot = get_angle_plot(signal, rx, 2)
    plt.gca().add_patch(angle_plot)
    k = signal.imag/signal.real
    x2 = (k*rx.imag+rx.real)/(1+k*k)
    y2 = k*x2
    plt.plot([rx.real, x2], [rx.imag, y2], 'b')
    x3 = x2 - 0.05
    y3 = y2

    x4 = (k*y3+x3)/(1+k*k)
    y4 = k*x4

    b = y2 + 1/k*x2
    x5 = (-1/k*y3+x3+(b*1/k))/(1+1/(k*k))
    y5 = b - 1/k*x5
    plt.plot([x3, x4], [y3, y4], 'b')
    plt.plot([x3, x5], [y3, y5], 'b')
    save_fig(plt, '../doc/image/sync_rx_phase_error_approx.svg')

    plt.clf()
    signal = np.exp(1j*(np.pi/4))
    plt.scatter(signal.real, signal.imag)
    signal2 = signal*np.exp(1j*(np.pi/2))
    plt.scatter(signal2.real, signal2.imag)
    plt.gca().annotate(r'', xy=(0, 0), xytext=(signal.real, signal.imag),
                       arrowprops=dict(arrowstyle='<-'))
    plt.gca().annotate(r'', xy=(0, 0), xytext=(signal2.real, signal2.imag),
                       arrowprops=dict(arrowstyle='<-'))
    plt.grid('on', ls=':')
    plt.xlabel('real')
    plt.ylabel('imag')
    rx = signal * np.exp(1j*(np.pi/8+np.pi/2))
    plt.gca().annotate(r'', xy=(0, 0), xytext=(rx.real, rx.imag),
                       arrowprops=dict(arrowstyle='<-'))
    plt.scatter(rx.real, rx.imag)
    plt.grid('on', ls=':')
    plt.xlabel('real')
    plt.ylabel('imag')
    plt.xlim([-1, 1])
    plt.ylim([0, 1])
    plt.legend(['true signal', 'estimated signal', 'received signal'])
    angle_plot = get_angle_plot(signal, rx, 0.5)
    plt.gca().add_patch(angle_plot)
    plt.gca().text(-0.1, 0.3, r'$\phi_e$')
    angle_plot = get_angle_plot(signal2, rx, 1)
    plt.gca().add_patch(angle_plot)
    plt.gca().text(-0.55, 0.3, r'$\hat{\phi}_e$')
    save_fig(plt, '../doc/image/sync_rx_phase_error_large.svg')

def plot_signal_freqz(d, fs, filename):
    plt.clf()
    plt.plot(np.linspace(0, fs/1e3, len(d), endpoint=False), 20*np.log10(np.abs(np.fft.fft(d))))
    plt.xlabel('f (kHz)')
    plt.ylabel('Magnitude (dB)')
    plt.tight_layout()
    plt.grid('on', ls='dotted')
    if filename:
        save_fig(plt, filename)

def sync_freq_demo():
    #### 1#hide
    # parameters
    Fsym = 1e6 # symbol rate
    UPS = 8 # up-sampling factor
    Fif = 2e6 # IF frequency
    # IF frequency delta at receiver side with frequency error
    Fif_prime = Fif + 20e3
    Fif_phi = 0.1

    Fs = Fsym*UPS # sampling frequency
    SNR = 30 # awgn (dB)

    # Modulation
    MODE = 4
    PHASE_OFFSET = np.pi/MODE
    # the maximum phase error without bit error
    PHASE_SPACE = 2*np.pi/MODE/2
    #mapping table for hard decision
    PHASE = 2*np.pi/8*np.array([1, 3, -3, -1])

    # True --> hard decision; False --> use tx data
    DECISION = False

    #### 2#hide
    # generate the PSK signal
    symbol = np.floor(MODE*np.random.rand(10000))
    signal = np.exp(1j*(2*np.pi/MODE*symbol + PHASE_OFFSET))

    # tx shaping filter
    fir_tx = rcosdesign(0.4, 5, UPS)

    s = np.kron(signal, np.concatenate((np.ones([1]), np.zeros([UPS-1]))))
    s = scipy.signal.lfilter(fir_tx, 1, s)
    # remove the filter delay
    s = s[(len(fir_tx)-1)//2:]
    plot_signal_freqz(s, Fs, '../doc/image/sync_demo_tx.png')#hide
    #### 3#hide
    # rx
    # modulation with IF to simulate a received signal after ADC
    r = np.real(s*np.exp(1j*2*np.pi*Fif/Fs*np.arange(len(s)))+Fif_phi)
    # add AWGN
    noise = np.random.normal(0, 1, len(r))
    noise *= 10**(-SNR/20)*np.sqrt(np.mean(r*r))*np.sqrt(UPS)
    r += noise
    plot_signal_freqz(r, Fs, '../doc/image/sync_demo_rx.png')#hide

    #### 3.1#hide
    # match filter
    fir_rx = rcosdesign(0.4, 3, UPS)
    r_buf = np.zeros(len(fir_rx), dtype=complex)

    # frequency synchronization
    q = np.zeros(np.ceil(len(s)/float(UPS)).astype(int), dtype=complex)
    q_est = np.zeros(np.ceil(len(s)/float(UPS)).astype(int), dtype=complex)
    freq_est = np.zeros(np.ceil(len(q)).astype(int))

    # estimated error
    ph_nco = 2*np.pi*(Fif_prime)/Fs
    freq_err = 0.0
    freq_err_i = 0.0
    freq_err_d = 0.0

    #### 4#hide
    index = 0
    # down sampling by a factor of UPS, then estimate the frequency error,
    # assume no clock error and sampling at best phase
    for i in range(0, len(r)):
        # step 1: NCO for frequency synchronization & frequency error compensation
        ph_nco = ph_nco - freq_err - 2*np.pi*(Fif_prime)/Fs
        if ph_nco > np.pi:
            ph_nco -= 2*np.pi
        elif ph_nco < -np.pi:
            ph_nco += 2*np.pi
        # step 2: down shift
        r_b = 2*r[i]*np.exp(1j*ph_nco)
        # step 3: filtering
        r_buf = np.roll(r_buf, 1)
        r_buf[0] = r_b
        r_b = np.sum(r_buf*fir_rx)

        # step 4: down-sampling by a factor of 8
        if i >= (len(fir_rx)-1)/2 and i%UPS == 0:
            # step 5: received signal
            z_in = r_b
            q_est[index] = z_in
            # step 6: decision
            if DECISION == 0:
                q_h = signal[index]
            else:
                # find the closest point on constellation
                p = np.arctan2(q_est[index].imag, q_est[index].real)
                b = np.argmin(np.abs(p-PHASE))
                q_h = np.exp(1j*PHASE[b])

            # step 7: estimate frequency error
            # remove the signal phase
            q[index] = q_est[index]*np.conj(q_h)
            # phase error ~ [-pi, pi]
            phi_e = np.arctan2(q[index].imag, q[index].real)

            # step 8: PI controller
            freq_err_i = freq_err_i + phi_e*0.01/UPS
            freq_err_d = phi_e*0.2/UPS
            freq_err = freq_err_i + freq_err_d
            freq_est[index] = freq_err
            index += 1
    ####
    # plot received signal
    figure(1)
    clf()
    # down shift
    rd = 2*r*np.exp(1j*(2*np.pi*-(Fif_prime)/Fs*np.arange(len(s))))
    z = scipy.signal.lfilter(fir_rx, 1, rd)
    # remove the filter delay
    z = z[(len(fir_rx)-1)//2:]

    plt.scatter(z[::UPS].real, z[::UPS].imag)
    plt.scatter(q_est[50:].real, q_est[50:].imag)
    plt.grid('on', ls=':')
    plt.xlabel('real')
    plt.ylabel('imag')
    plt.legend(['without freq sync', 'with freq sync'])
    save_fig(plt, '../doc/image/sync_demo_scatter.png')
    # estimated phase error
    plt.figure(2)
    plt.clf()
    plt.plot(freq_est[:index])
    plt.xlabel('n (samples)')
    plt.ylabel('e[n]')
    plt.title('Estimated Phase Error')
    plt.grid('on', ls=':')
    save_fig(plt, '../doc/image/sync_demo_err.png')

def plot_sync_clock():
    def add_annotate(x, y, offset, txt=None):
        ant = plt.gca().annotate("", xy=(x, y), xytext=offset,
                textcoords='offset points', ha='right', va='bottom',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        if txt is None:
            txt = '%0.2f'%(y)
        ant.set_text(txt)

    def sample_phase(smp, ofst, filename):
        fir_tx = rcosdesign(0.4, 5, 20, 'normal')
        fir_tx = fir_tx/max(fir_tx)
        plt.clf()
        D = (len(fir_tx)-1)/2
        x = np.arange(len(fir_tx)) - D - smp
        l1 = plt.plot(x/20., fir_tx)
        l3 = plt.plot(x/20.-1, fir_tx)
        xm = np.array([-ofst, 0])
        ym = [fir_tx[smp-ofst+D], fir_tx[smp+20+D]]
        m1 = plt.plot(xm/20.0, ym, 'bs')
        m1[0].set_color(l1[0].get_color())
        plt.plot([0, 0], [-0.2, 1.2], 'b-.')
        plt.plot([-ofst/20., -ofst/20.], [-0.2, 1.2], 'b-.')

        add_annotate(0, fir_tx[smp+D+20], (-20, 0))
        add_annotate(-ofst/20., fir_tx[smp+D-ofst], (-20, 0))
        plt.grid('on', ls=":")
        plt.xlabel('n (symbols)')
        plt.xlim([-3, 3])
        plt.ylim([-0.2, 1.2])
        save_fig(plt, filename)

    fir_tx = rcosdesign(0.4, 5, 20, 'normal')
    fir_tx = fir_tx/max(fir_tx)
    plt.clf()
    D = (len(fir_tx)-1)/2
    smp = 5
    ofst = 20
    x = (np.arange(len(fir_tx)) - D - smp)
    l1 = plt.plot(x/20.0, fir_tx)
    l2 = plt.plot(x/20.0+1, fir_tx)
    l3 = plt.plot(x/20.0-1, fir_tx)
    xm = np.array([-ofst, 0, ofst])
    plt.plot(xm/20.0, fir_tx[smp+xm+D], 'bo')
    for x in xm:
        plt.plot([x/20., x/20.], [-0.2, 1.2], 'b-.')
    plt.plot([-smp/20., -smp/20.], [-0.2, 1.2], 'b-.')
    gca().annotate(r'', xy=(0, 1), xytext=(-smp/20.0, 1), arrowprops=dict(arrowstyle='<->'))
    gca().annotate(r'$\tau$', xy=(0, 1), xytext=(-smp/20.0, 1.05))
    plt.grid('on', ls=":")
    plt.xlabel('n (symbols)')
    xv = np.array([0, -20, 20, -40]) + smp + D
    add_annotate(0, fir_tx[smp+D], (-20, 0), r'$f(\tau)$')
    add_annotate(-ofst/20.0, fir_tx[smp+D-ofst], (-20, 0), r'$f(\tau-1)$')
    add_annotate(+ofst/20.0, fir_tx[smp+D+ofst], (50, 10), r'$f(\tau+1)$')
    plt.xlim([-3, 3])
    plt.ylim([-0.2, 1.2])
    save_fig(plt, '../doc/image/sync_clk_mm.svg')
    sample_phase(0, 20, '../doc/image/sync_clk_mm_optimal.svg')
    sample_phase(5, 20, '../doc/image/sync_clk_mm_late.svg')
    sample_phase(-5, 20, '../doc/image/sync_clk_mm_early.svg')
    sample_phase(-5, 15, '../doc/image/sync_clk_mm_fast.svg')
    sample_phase(5, 25, '../doc/image/sync_clk_mm_slow.svg')

    plt.clf()
    plt.stem(np.arange(-5, 5), np.random.rand(10), basefmt='b')
    plt.plot([smp/20., smp/20.], [-0.2, 1.2], 'r-.')
    add_annotate(smp/20., 0, (20, 0), r'$\tau$')
    plt.ylim([-0.2, 1.2])
    x = np.arange(len(fir_tx)) - D + smp
    plt.grid('on', ls=":")
    plt.xlabel('samples')
    save_fig(plt, '../doc/image/sync_clk_cmp.svg')
    l1 = plt.plot(x/20.0, fir_tx)
    save_fig(plt, '../doc/image/sync_clk_cmp2.svg')

    plt.clf()
    signal = np.ones(100)
    signal[::2] = -1
    # shaping filter
    fir_tx = rcosdesign(0.4, 3, 24, 'normal')
    fir_tx = fir_tx/max(fir_tx)
    s = np.kron(signal, np.concatenate((np.ones([1]), np.zeros([24-1]))))
    s = scipy.signal.lfilter(fir_tx, 1, s)
    # remove the filter delay
    s = s[(len(fir_tx)-1)/2:]
    plt.plot(np.arange(0, len(s))/24., s)
    plt.stem(np.arange(0, len(s), 24)/24., s[0::24], 'b-.', basefmt=' ')
    plt.stem(np.arange(5, len(s), 24)/24., s[5::24], 'r-.', basefmt=' ')
    plt.grid('on', ls=':')
    plt.xlim([1840/24., 1960/24.])
    plt.xlabel('symbols')
    save_fig(plt, '../doc/image/sync_clk_bpsk.svg')
    plt.clf()
    signal = np.floor(np.random.rand(100)+0.5)*2-1
    s = np.kron(signal, np.concatenate((np.ones([1]), np.zeros([24-1]))))
    s = scipy.signal.lfilter(fir_tx, 1, s)
    plt.plot(np.arange(0, len(s))/24., s, 'b-o')
    plt.plot(np.arange(12, len(s), 24)/24., s[12::24], 'ro')
    plt.stem(np.arange(0, len(s), 24)/24., s[0::24], 'b-.', basefmt=' ')
    plt.grid('on', ls=':')
    plt.xlabel('symbols')
    save_fig(plt, '../doc/image/sync_clk_bpsk_tr.svg')

    plt.clf()
    x = np.arange(len(fir_tx)) - len(fir_tx)/2
    plt.plot(x/24., fir_tx)
    x = np.array([-15, 0, 15])
    plt.stem(x/24., fir_tx[x+len(fir_tx)/2], 'b-.', basefmt=' ')
    add_annotate(x[0]/24., fir_tx[x[0]+len(fir_tx)/2], (40, -20), r'$r_{early}$')
    add_annotate(x[1]/24., fir_tx[x[1]+len(fir_tx)/2], (0, -20), r'$r_{on}$')
    add_annotate(x[2]/24., fir_tx[x[2]+len(fir_tx)/2], (-20, -20), r'$r_{late}$')
    plt.grid('on', ls=':')
    plt.xlabel('symbols')
    plt.xlim([-25/24., 25/24.])
    save_fig(plt, '../doc/image/sync_clk_bpsk_opt.svg')
    plt.clf()
    x = np.arange(len(fir_tx)) - len(fir_tx)/2
    plt.plot(x/24., fir_tx)
    x = np.array([-15, 0, 15])-5
    plt.stem(x/24., fir_tx[x+len(fir_tx)/2], 'b-.', basefmt=' ')
    add_annotate(x[0]/24., fir_tx[x[0]+len(fir_tx)/2], (40, -20), r'$r_{early}$')
    add_annotate(x[1]/24., fir_tx[x[1]+len(fir_tx)/2], (0, -20), r'$r_{on}$')
    add_annotate(x[2]/24., fir_tx[x[2]+len(fir_tx)/2], (-20, -20), r'$r_{late}$')
    plt.grid('on', ls=':')
    plt.xlabel('symbols')
    plt.xlim([-25/24., 25/24.])
    save_fig(plt, '../doc/image/sync_clk_bpsk_early.svg')

    plt.clf()
    x = np.arange(len(fir_tx)) - len(fir_tx)/2
    plt.plot(x/24., fir_tx)
    x = np.array([-15, 0, 15])+5
    plt.stem(x/24., fir_tx[x+len(fir_tx)/2], 'b-.', basefmt=' ')
    add_annotate(x[0]/24., fir_tx[x[0]+len(fir_tx)/2], (40, -20), r'$r_{early}$')
    add_annotate(x[1]/24., fir_tx[x[1]+len(fir_tx)/2], (0, -20), r'$r_{on}$')
    add_annotate(x[2]/24., fir_tx[x[2]+len(fir_tx)/2], (-20, -20), r'$r_{late}$')
    plt.grid('on', ls=':')
    plt.xlabel('symbols')
    plt.xlim([-25/24., 25/24.])
    save_fig(plt, '../doc/image/sync_clk_bpsk_late.svg')
# sampling rate conversion
def rx_src(in_data, in_sfreq, out_sfreq):
    if in_sfreq == out_sfreq:
        return in_data
    ORDER = 16 # the filter order in Fs(here 76kHz) sampling rate
    M = 20 # up sampling factor
    N = in_sfreq/out_sfreq*M # down sampling factor
    # low pass filter filter
    F = max(N, M)
    flt = scipy.signal.firwin2(ORDER*M+1, [0, 0.5/F, 1.0/F, 1], [1, 1, 0, 0])
    delay = int((len(flt)-1)/2)
    # only save the right half coefficients
    flt = flt[delay:]
    flt = np.hstack((flt, np.zeros(M*2)))

    # allocate the memory for result
    out = np.zeros(np.ceil(len(in_data)/N*M).astype(int), dtype=in_data.dtype)
    idx_out = 0
    # initialize value
    alpha = N-M
    in_buf = np.zeros(ORDER+1, dtype=in_data.dtype)
    for i in range(ORDER//2):
        in_buf[ORDER//2-1-i] = in_data[i]

    for i in range(ORDER//2, len(in_data)):
        # shift in one data
        in_buf = np.roll(in_buf, 1)
        in_buf[0] = in_data[i]
        alpha = alpha + M
        # need output?
        while alpha >= N:
            delta = alpha - N
            #delta = M - alpha
            output = 0
            # left side
            for k in range(ORDER//2+1):
                delta_i = int(delta+k*M)
                delta_f = delta+k*M - delta_i
                coeff_k = flt[delta_i]*(1-delta_f) + flt[delta_i+1]*delta_f
                output = output + coeff_k*in_buf[ORDER//2-k]

            #right side
            delta2 = M - delta
            for k in range(ORDER//2):
                delta_i = int(delta2+k*M)
                delta_f = delta2+k*M - delta_i
                coeff_k = flt[delta_i]*(1-delta_f) + flt[delta_i+1]*delta_f
                output = output + coeff_k*in_buf[ORDER//2+1+k]

            #output
            out[idx_out] = output*M
            idx_out += 1
            alpha -= N

    out = out[:idx_out]
    return out

def rx_src_test():
    #### 1#hide
    s = np.cos(2*np.pi*1e6/24e6*np.arange(10000))
    s2 = rx_src(s, 24e6, 24e6*(1+100e-6))
    #### 2#hide
    s3 = rx_src(s2, 24e6*(1+100e-6), 24e6)
    np.mean(np.abs(s[100:len(s3)]-s3[100:]))#eval
    #### 3#hide
    plt.clf()
    plt.subplot(121)
    x = np.arange(10, 40)
    plt.plot(x, s[x])
    plt.plot(x, s2[x])
    plt.xlabel('n')
    plt.grid('on', ls=':')
    plt.legend(['s', 's2'])
    plt.subplot(122)
    x = x + 9937
    plt.plot(x, s[x])
    plt.plot(x, s2[x])
    plt.xlabel('n')
    plt.grid('on', ls=':')
    plt.legend(['s', 's2'])
    save_fig(plt, '../doc/image/sync_rx_src_test.svg')

def sync_clk_demo():
    #### 1#hide
    Fsym = 1e6 # symbol rate
    UPS = 8 # up sampling factor
    Fs = Fsym*UPS # tx sampling rate
    Fs_rx = Fs*(1+100e-6) # rx sampling rate with clock offset
    DECISION = False # True --> hard decision, False --> use true signal
    SNR = 30 # awgn (dB)
    # Modulation
    MODE = 4
    PHASE_OFFSET = np.pi/MODE
    # the maximum phase error without bit error
    PHASE_SPACE = 2*np.pi/MODE/2
    #mapping table for hard decision
    PHASE = 2*np.pi/8*np.array([1, 3, -3, -1])

    #### 2#hide
    symbol = np.floor(MODE*np.random.rand(10000))
    signal = np.exp(1j*(2*np.pi/MODE*symbol + PHASE_OFFSET))

    # tx shaping filter
    fir_tx = rcosdesign(0.4, 3, UPS)
    s = np.kron(signal, np.concatenate((np.ones([1]), np.zeros([UPS-1]))))
    s = scipy.signal.lfilter(fir_tx, 1, s)
    # remove the filter delay
    s = s[(len(fir_tx)-1)//2:]

    #### 3#hide
    # rx
    # add timing error
    r = rx_src(s, Fs, Fs_rx)
    # add AWGN
    noise = np.random.normal(0, 0.5, len(r)*2)
    noise = noise[::2] + 1j*noise[1::2]
    noise *= 10**(-SNR/20)*np.sqrt(np.mean(r*np.conj(r)).real)*np.sqrt(UPS)
    r += noise

    #### 3.1#hide
    # match filter
    fir_rx = rcosdesign(0.4, 3, UPS)
    z = scipy.signal.lfilter(fir_rx, 1, r)
    # remove the filter delay
    z = z[(len(fir_rx)-1)//2:]

    #### 4#hide
    # estimated error
    timing_err = 0.0
    timing_err_p = 0.0
    timing_err_i = 0.0
    # hard & soft values and phase error of previous symbol
    z_h_p, z_s_p, ph_e_p = 0, 0, 0

    # index & counter for down-sampling by a factor of UPS
    ds_idx, ds_cnt = 0, UPS-1

    # timing recovery parameters
    ORDER = 16 #the filter order in Fs sampling rate
    M = 20 # up sampling factor
    N = M # down sampling factor
    flt = scipy.signal.firwin2(ORDER*M+1, [0, 0.5/M, 1.0/M, 1], [1, 1, 0, 0])
    # only saving the right half coefficients
    flt = flt[ORDER*M//2:]
    flt = np.hstack((flt, np.zeros(M+1)))
    # timing recovery block initialize value
    alpha = 2*M
    z_buf = np.zeros(ORDER+1, dtype=z.dtype)
    for i in range(ORDER//2):
        z_buf[ORDER//2-1-i] = z[i]

    #### 5#hide
    # output from the timing recovery block
    z2 = np.zeros(np.ceil(len(z)*(1+200e-6)/N*M).astype(int), dtype=z.dtype)
    # error display
    ph_err = np.zeros(np.ceil(float(len(z))/UPS).astype(int), dtype=z.dtype)
    terr_est = np.zeros(np.ceil(len(ph_err)).astype(int))
    # symbol after timing recovery
    z_out = np.zeros(np.ceil(len(ph_err)).astype(int), dtype=z.dtype)

    # the current sample index from the timing recovery block
    idx_tr = 0
    # the index of next sample from the timing recovery block to be processed
    idx_tr_p = 0
    #### 6#hide
    for i in range(ORDER//2, len(z)):
        # shift in one data for timing recovery
        z_buf = np.roll(z_buf, 1)
        z_buf[0] = z[i]
        # update the alpha
        alpha -= M
        # timing recovery
        while alpha <= M:
            output = 0
            # left side
            delta = M - alpha
            for k in range(ORDER//2+1):
                delta_i = int(delta+k*M)
                delta_f = delta+k*M - delta_i
                coeff_k = flt[delta_i]*(1-delta_f) + flt[delta_i+1]*delta_f
                output += coeff_k*z_buf[ORDER//2-k]

            # right side
            delta2 = M-delta
            for k in range(ORDER//2):
                delta_i = int(delta2+k*M)
                delta_f = delta2+k*M - delta_i
                coeff_k = flt[delta_i]*(1-delta_f) + flt[delta_i+1]*delta_f
                output += coeff_k*z_buf[ORDER//2+1+k]

            #output
            z2[idx_tr] = output*M
            idx_tr += 1

            # next output position
            alpha += N
            # compensate the timing error
            alpha += timing_err

        # down sampling by factor UPS, calculate the timing error,
        while idx_tr_p < idx_tr:
            # new input from the sampling rate conversion block
            z_in = z2[idx_tr_p]
            idx_tr_p += 1
            ds_cnt += 1
            # sampling for hard decision and timing error estimation
            if ds_cnt >= UPS:
                ds_cnt -= UPS
                # hard decision
                if DECISION == 0:
                    z_h = signal[ds_idx]
                else:
                    ph = np.arctan2(z_in.imag, z_in.real)
                    b = np.argmin(np.abs(ph-PHASE))
                    z_h = np.exp(1j*PHASE[b])

                # timing error calculating
                e_c = z_in*np.conj(z_h)
                ph_e = np.arctan2(e_c.imag, e_c.real)
                # calibrate
                if abs(ph_e_p) > PHASE_SPACE or abs(ph_e) > PHASE_SPACE:
                    tmer = 0
                else:
                    tmerr = np.real(np.conj(z_h_p)*z_in - np.conj(z_h)*z_s_p)
                # for next symbol
                z_s_p, z_h_p, ph_e_p = z_in, z_h, ph_e
                # PI controller
                timing_err_i += tmerr*0.001
                timing_err_p = tmerr*0.09
                timing_err = timing_err_i + timing_err_p
                # for plot #hide
                terr_est[ds_idx] = timing_err #hide
                z_out[ds_idx] = z_in #hide
                ph_err[ds_idx] = e_c #hide
                ds_idx += 1

    #### figure
    plt.figure(1)
    plt.clf()
    plt.scatter(z[::UPS].real, z[::UPS].imag)
    plt.grid('on', ls=':')
    save_fig(plt, '../doc/image/sync_clk_demo_r.png')

    plt.figure(2)
    plt.clf()
    plt.plot(terr_est[:ds_idx])
    plt.plot([0, ds_idx-1], [100e-6*20, 100e-6*20])
    plt.grid('on', ls=':')
    plt.xlabel('symbol')
    plt.ylabel('timing err')
    plt.legend(['estimated timing err', 'timing err'])
    plt.ylim([-0.04, 0.04])
    save_fig(plt, '../doc/image/sync_clk_demo_err.png')
    ph_err = ph_err[:ds_idx] # remove the empty tail
    # measurement
    e = ph_err[1:] - ph_err[:-1]
    q = ph_err[1:]
    ##
    IGNORE = 300
    z_out = z_out[IGNORE:ds_idx]
    plt.figure(3)
    plt.clf()
    plt.scatter(z_out.real, z_out.imag)
    plt.grid('on', ls=':')
    save_fig(plt, '../doc/image/sync_clk_demo_sym.png')

    # devm parameters
    rms_devm_thrd = 0.2
    peak_devm_thrd = 0.35
    devm_99_thrd = 0.3

    ##
    e = e[IGNORE:]
    q = q[IGNORE:]
    hc = np.histogram(np.abs(e/q), [0, devm_99_thrd])
    devm_99 = hc[0][0]/len(q)*100
    rms_devm = np.sqrt(np.mean(e*np.conj(e))/np.mean(q*np.conj(q))).real
    peak_dvem = np.sqrt(np.max(e*np.conj(e)) /(np.mean(q*np.conj(q)))).real

    if rms_devm > rms_devm_thrd:
        print('RMS DEVEM:%f, failed'%rms_devm)
    else:
        print('RMS DEVEM:%f, passed'%rms_devm)

    if peak_dvem > peak_devm_thrd:
        print('PEAK DEVEM:%f failed'%peak_dvem)
    else:
        print('PEAK DEVEM:%f passed'%peak_dvem)

    if devm_99 < 99:
        print('99 DEVEM:%f failed'%devm_99)
    else:
        print('99 DEVEM:%f passed'%devm_99)

def sync_clk_bpsk(signal, ups, dis=1):
    PERIOD = ups  # upsample factor
    sample_ins = dis + 8 #initial sample position
    # output
    smp_tm = np.zeros(np.ceil(len(signal)/float(PERIOD)).astype(int)+1)
    msg = np.zeros(np.ceil(len(signal)/float(PERIOD)).astype(int)+1)

    ds_cnt = 0 # down-sampling counter
    sym_cnt = 0 # current symbol index after down-sampling
    for i in range(int(np.floor((len(signal)))-dis)):
        ds_cnt += 1
        if ds_cnt >= sample_ins:
            ds_cnt = 0
            s_early, s_on, s_late = signal[[i-dis, i, i+dis]]
            if (s_on - s_early)*(s_on - s_late) > 0:
                # on time sampling
                sample_ins = PERIOD
            elif s_on*(s_on - s_early) > 0 and s_on*(s_on - s_late) < 0:
                # early sampling, delay the next one
                sample_ins = PERIOD + 1
            elif s_on*(s_on - s_early) < 0 and s_on*(s_on - s_late) > 0:
                # late sampling, advance the next one
                sample_ins = PERIOD - 1
            smp_tm[sym_cnt] = i
            # decision slicer
            msg[sym_cnt] = signal[i] > 0
            sym_cnt += 1
    return (msg.astype(int), smp_tm.astype(int))

def sync_lfsr(ini=None, gen=None):
    if ini is None:
        ini = np.array([1, 0, 0, 0, 0, 0, 0])
    if gen is None:
        gen = np.array([1, 1, 0, 0, 0, 0, 0, 1])
    pn = np.zeros(2**len(ini)-1)
    for i in range(len(pn)-1):
        out = np.mod(np.sum(ini*gen[1:]), 2)
        pn[i] = out
        ini = np.roll(ini, 1)
        ini[0] = out
    return pn.astype(int)

def sync_clk_bpsk_demo():
    #### 1#hide
    # parameters
    Fs = 152000 # sample frequency(Hz)
    Fd = 1187.5 # data rate(Hz)
    CLK_OFFSET = 200e-6 # (+/-200ppm)
    SNR = 7 # in dB
    #### 2#hide
    msg_bits = np.random.randint(0, 2, 1000)
    # add PN sequence for frame synchronization
    pn = sync_lfsr()
    msg_cyc = np.hstack((pn, msg_bits, np.zeros(5, dtype=int)))

    #### 3#hide
    # differential encoding
    bit_p = 0
    for i in range(len(msg_cyc)):
        msg_cyc[i] = np.bitwise_xor(bit_p, msg_cyc[i])
        bit_p = msg_cyc[i]

    #### 4#hide
    # BPSK modulation
    # in real case this may be replaced by a lookup table, rather than a filter
    # generate the shaping filter
    UP_SAMP = int(Fs/Fd)
    UP_BASE = UP_SAMP//2
    hf = np.zeros(np.ceil(UP_SAMP/2*UP_BASE/2).astype(int))
    hf[:UP_BASE] = np.cos(np.pi/4*np.linspace(0, 2, UP_BASE))
    hf = np.hstack((hf, hf[-1:0:-1]))
    ht = np.fft.ifft(hf)
    ht = np.hstack((ht[-1 - (len(ht)-1)//2-1-1:], ht[0:(len(ht)-1)//2]))
    # hard coded here, 401 taps around center, just for simulation
    ht = ht[2048-200:2048+201].real

    #### 5#hide
    # generate bi-phase symbol
    msg_biphase = np.kron(msg_cyc*2-1, [1, -1])
    # shaping filter
    u = np.zeros(UP_BASE)
    u[0] = 1
    msg_shaping = scipy.signal.lfilter(ht, 1, np.kron(msg_biphase, u))*UP_BASE

    #### 6#hide
    # add clock error
    msg_signal = rx_src(msg_shaping, Fs, Fs*(1+CLK_OFFSET))
    # add noise
    gain = 1.0/(np.sum(msg_signal*msg_signal)/len(msg_signal))/(Fs/Fd/4)
    msg_signal = np.sqrt(gain)*msg_signal
    noise = np.random.normal(0, 1, len(msg_signal))
    noise *= 10**(-SNR/20)*np.sqrt(np.mean(msg_signal*msg_signal))*np.sqrt(UP_BASE)
    msg_signal += noise

    #### 7#hide
    # BSKP demodulation
    # down sampling by factor 8
    rx_sig = msg_signal
    fir_ds8 = scipy.signal.firwin2(40, [0, 4750./(Fs), 14250./(Fs), 1], [1, 1, 0, 0])
    rx_sig = scipy.signal.lfilter(fir_ds8, 1, rx_sig)
    rx_sig = rx_sig[::8]
    #### 8#hide
    # generate the matching filter
    hf8 = np.zeros(int(UP_SAMP/8/2*UP_BASE/2))
    hf8[:UP_BASE] = np.cos(np.pi/4*np.linspace(0, 2, UP_BASE))
    hf8 = np.hstack((hf8, hf8[-1:0:-1]))
    ht8 = np.fft.ifft(hf8)
    ht8 = np.hstack((ht8[-1- (len(ht8)-1)//2-1-1:-1], ht8[:(len(ht8)-1)//2]))
    # filter the signal from bit '1'
    s = np.hstack((np.ones(8), -1*np.ones(8), np.zeros(len(ht8))))
    fir_match = scipy.signal.lfilter(ht8, 1, s)
    fir_match = fir_match[246:285].real
    # match filtering
    rx_sig = scipy.signal.lfilter(fir_match, 1, rx_sig)

    #### 9#hide
    # Timing recovery and BPSK demodulation
    rx_bits, smp_time = sync_clk_bpsk(rx_sig, UP_SAMP/8, 1)

    #### 10#hide
    # differential decoding
    rx_bits_diff = np.bitwise_xor(np.hstack(([0], rx_bits[:-1])), rx_bits)

    # frame synchronization
    sync_corr = scipy.signal.convolve(rx_bits_diff*2-1, np.flipud(pn)*2-1)
    start = np.where(sync_corr > 80)[0]
    if len(start) == 0:
        print('Frame Header error!')
    else:
        print('Frame Header: %d at %d'%(sync_corr[start[0]], start[0]))
        msg_rx = rx_bits_diff[start[0]+1:]
        lenmin = min(len(msg_bits), len(msg_rx))
        err = sum(abs(msg_bits[:lenmin] - msg_rx[:lenmin]))
        print('BER: %f (%d/%d)'%(float(err)/lenmin, err, lenmin))

    ####
    plt.clf()
    plt.subplot(121)
    plt.plot(ht)
    plt.grid('on', ls=":")
    plt.xlabel('n (samples)')
    plt.ylabel('$h_T(n)$')
    plt.subplot(122)
    plt.plot(np.linspace(0, Fs, len(hf)), hf)
    plt.grid('on', ls=":")
    plt.xlim([0, Fd*3])
    plt.xlabel('f (Hz)')
    plt.ylabel('$H_T(f)$')
    save_fig(plt, '../doc/image/sync_clk_bpsk_shaping.svg')

    plt.clf()
    s = np.hstack((np.ones(64), -1*np.ones(64), np.zeros(len(ht))))
    sym = scipy.signal.lfilter(ht, 1, s)
    x = np.arange(len(ht)/2-50, len(ht)/2+200)
    plt.plot(x-266, sym[x])
    plt.plot(x-266, -sym[x])
    plt.grid('on', ls=":")
    plt.legend(['bit 1', 'bit 0'])
    plt.xlabel('n (samples)')
    save_fig(plt, '../doc/image/sync_clk_bpsk_biphase.svg')

    plt.clf()
    plt.plot(rx_sig)
    plt.plot(smp_time, rx_sig[smp_time], 'o')
    plt.grid('on', ls=":")
    plt.xlim([0, 400])
    plt.xlabel('n (samples)')
    save_fig(plt, '../doc/image/sync_clk_bpsk_sampling.png')

def sync_phase_est_test():
    #### 1#hide
    alpna = np.linspace(0, np.pi/4, 10000)
    a = np.cos(alpha)
    b = np.sin(alpha)
    alpha_e = (b/a)/(1+0.2815*(b/a)**2)

    plt.clf()
    plt.plot(alpha*180/np.pi, (alpha-alpha_e)*180/np.pi)
    plt.grid('on', ls=':')
    plt.xlabel('angle (in degree)')
    plt.ylabel('estimation error (in degree)')
    save_fig(plt, '../doc/image/sync_clk_phase_est.png')#hide
