import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal
import scipy.io
def integral_up_sampling_test():
    #### 1 #hide
    # signal parameters
    f = 20e3
    fs = 1e6
    # generage the sinusoid signal
    s = np.sin(2*np.pi*f/fs*np.arange(1000))
    # plot the signal
    plt.clf()#hide
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, len(s))/fs*1e3, s)
    plt.xlabel('t(ms)')
    plt.grid('on', ls='dotted')
    plt.subplot(2, 1, 2)
    # plot the frequency response
    plt.plot(np.linspace(0, fs/1e3, len(s), endpoint=False), 20*np.log10(abs(np.fft.fft(s))))
    plt.xlabel('$f$ (kHz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid('on', ls='dotted')
    plt.tight_layout()
    save_fig(plt, "image/up_sampling_demo_signal.svg")#hide
    plt.show()#noexec

    #### 2 #hide
    plt.clf() #hide
    # the new sampling frequency
    fo = 3*fs
    # up-sampling
    s3 = np.kron(s, [1, 0, 0])
    # plot the signal after up-sampling
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, len(s3))/(fo)*1e3, s3, '.')
    plt.xlim([0, 25/fs*1e3])
    plt.xlabel('t(ms)')
    plt.grid('on', ls='dotted')
    plt.subplot(2, 1, 2)
    plt.plot(np.linspace(0, fo/1e3, len(s3), endpoint=False), 20*np.log10(abs(np.fft.fft(s3))))
    plt.xlabel('$f$ (kHz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid('on', ls='dotted')
    plt.tight_layout()
    save_fig(plt, "image/up_sampling_demo_fo.svg")#hide
    plt.show()#noexec

    ####  3 filter #hide
    # design a low-pass filter
    N = 21
    b = scipy.signal.firwin(N, fs/fo)
    # filtering
    y = 3*scipy.signal.lfilter(b, 1, s3)
    # plot the signal after filtering
    plt.clf() #hide
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, len(s))/fs*1e3, s, 'ro')
    plt.plot(np.arange(0, len(y)-N/2)/(fo)*1e3, y[N/2:], '.')
    plt.xlim([0, 25/fs*1e3])
    plt.xlabel('t(ms)')
    plt.grid('on', ls='dotted')
    plt.legend(['original signal', 'up-sampling by a factor of 3'])
    plt.subplot(2, 1, 2)
    plt.plot(np.linspace(0, fo/1e3, len(s3), endpoint=False), 20*np.log10(abs(np.fft.fft(y))))
    plt.xlabel('$f$ (kHz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid('on', ls='dotted')
    plt.tight_layout()
    save_fig(plt, "image/up_sampling_demo_filter.svg")#hide
    plt.show()#noexec

    #### 4 insert 1#hide
    plt.clf() #hide
    # the new sampling frequency
    s3 = scipy.signal.lfilter([1, 1, 1], 1, np.kron(s, [1, 0, 0]))
    # plot the signal after up-sampling
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, len(s3))/(fo)*1e3, s3, '.')
    plt.xlim([0, 25/fs*1e3])
    plt.xlabel('t(ms)')
    plt.grid('on', ls='dotted')
    plt.subplot(2, 1, 2)
    plt.plot(np.linspace(0, fo/1e3, len(s3), endpoint=False), 20*np.log10(abs(np.fft.fft(s3))))
    w, h = scipy.signal.freqz([1, 1, 1])
    plt.plot(w/np.pi*fo/1e3, 20*np.log10(abs(h)))
    plt.xlabel('$f$ (kHz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid('on', ls='dotted')
    plt.tight_layout()
    plt.legend(['up-sampling output', 'filter [1, 1, 1]'])
    save_fig(plt, "image/up_sampling_insert1.svg")#hide
    plt.show()#noexec

def signal_freqz(d, fs, filename):
    plt.clf()
    plt.subplot(211)
    plt.plot(np.arange(0, len(d))/fs*1e3, d)
    plt.grid('on', ls='dotted')
    plt.xlabel('t(ms)')
    plt.subplot(212)
    plt.plot(np.linspace(0, fs/1e3, len(d), endpoint=False), 20*np.log10(np.abs(np.fft.fft(d))))
    plt.xlabel('f (kHz)')
    plt.ylabel('Magnitude (dB)')
    plt.tight_layout()
    plt.grid('on', ls='dotted')
    if filename:
        save_fig(plt, filename)

def src_step_by_step(d):
    d12n = d
    clf()
    stem(d12n[:121], basefmt='')
    ax1 = gca()
    for i in range(0, 121, 12):
        ax1.annotate('d[%d]'%(i/12), xy=(i, d12n[i]), xytext=(i+2, d12n[i]))

    xlabel('n (sampling frequency = M$f_i$)')
    ax1.set_ylabel('d12')
    #ax1.set_ylim([-1500, 5500])
    grid('on', ls='dotted')
    save_fig(plt, "image/fsrc_step0.svg")#hide

    ax2 = gca().twinx()
    ax2.plot(flt, 'r')
    ax2.plot([12*5, 12*5], [0, 0.06], 'g--')
    ax2.set_ylim([-0.015, 0.06])
    ax2.set_ylabel('filter coefficients')
    ax2.annotate(r'$\alpha=19$', xy=(12*5, flt[12*5]), xytext=(12*5+2, flt[12*5]))
    save_fig(plt, "image/fsrc_step1.svg")#hide

    plot(np.arange(0, len(flt), 12), flt[0::12], 'gs')
    save_fig(plt, "image/fsrc_step2.svg")#hide

    clf()
    stem(d12n[:121], basefmt='')
    ax1 = gca()
    for i in range(0, 121, 12):
        ax1.annotate('d[%d]'%(i/12), xy=(i, d12n[i]), xytext=(i+2, d12n[i]))

    xlabel('n (sampling frequency = M$f_i$)')
    ax1.set_ylabel('d12')
    grid('on', ls='dotted')
    ax1.plot([12*5, 12*5], [0, 5500], 'r--')
    ax1.annotate(r'$\alpha=0$', xy=(12*5, 5500), xytext=(12*5+2, 5500))
    save_fig(plt, "image/fsrc_step2_2.svg")#hide

    clf()
    stem(d12n[12:121+12], basefmt='')
    ax1 = gca()
    for i in range(12, 121+12, 12):
        ax1.annotate('d[%d]'%(i/12), xy=(i-12, d12n[i]), xytext=(i-12+2, d12n[i]))
    xlabel('n (sampling frequency = M$f_i$)')
    ax1.set_ylabel('d12')
    grid('on', ls='dotted')
    ax1.plot([12*5, 12*5], [0, 5500], 'g--')
    ax1.plot([12*4, 12*4], [0, 5500], 'r--')
    ax1.annotate(r'$\alpha=12$', xy=(12*5, 5000), xytext=(12*4+2, 5500))
    ax1.annotate(r'', xy=(12*4, 5000), xytext=(12*5, 5000), arrowprops=dict(arrowstyle='<->'))
    save_fig(plt, "image/fsrc_step3.svg")#hide


    clf()
    n = 24
    stem(d12n[n:121+n], basefmt='')
    ax1 = gca()
    for i in range(n, 121+n, 12):
        ax1.annotate('d[%d]'%(i/12), xy=(i-n, d12n[i]), xytext=(i-n+2, d12n[i]))
    xlabel('n (sampling frequency = M$f_i$)')
    ax1.set_ylabel('d12')
    grid('on', ls='dotted')
    ax1.plot([12*5, 12*5], [0, 5500], 'g--')
    ax1.plot([12*3, 12*3], [0, 5500], 'r--')
    ax1.annotate(r'$\alpha=%d$'%n, xy=(12*5, 5500), xytext=(12*3+2, 5500))
    ax1.annotate(r'', xy=(12*3, 5000), xytext=(12*5, 5000), arrowprops=dict(arrowstyle='<->'))
    save_fig(plt, "image/fsrc_step4.svg")#hide

    ax1.plot([12*3+19, 12*3+19], [0, 5500], 'b--')
    ax1.annotate(r'$\delta=\alpha-19=%d$'%(n-19), xy=(12*5, 5500), xytext=(12*3+19+1, 5500))
    ax1.annotate(r'', xy=(12*3+19, 5300), xytext=(12*5, 5300), arrowprops=dict(arrowstyle='<->'))

    ax2 = gca().twinx()
    ax2.plot(flt[5:], 'r')
    ax2.set_ylim([-0.0195, 0.06])
    plot(np.arange(0, len(flt)-1, 12), flt[5::12], 'gs')
    save_fig(plt, "image/fsrc_step5.svg")#hide

    clf()
    n = 24
    stem(d12n[n:121+n], basefmt='')
    ax1 = gca()
    for i in range(n, 121+n, 12):
        ax1.annotate('d[%d]'%(i/12), xy=(i-n, d12n[i]), xytext=(i-n+2, d12n[i]))
    xlabel('n (sampling frequency = M$f_i$)')
    ax1.set_ylabel('d12')
    grid('on', ls='dotted')
    ax1.plot([12*5, 12*5], [0, 5500], 'g--')
    ax1.plot([12*3+19, 12*3+19], [0, 5500], 'r--')
    ax1.annotate(r'$\alpha=%d$'%(n-19), xy=(12*5, 5500), xytext=(12*3+19+1, 5500))
    ax1.annotate(r'', xy=(12*3+19, 5000), xytext=(12*5, 5000), arrowprops=dict(arrowstyle='<->'))
    save_fig(plt, "image/fsrc_step6.svg")#hide

    clf()
    n = 12
    stem(d12n[n:121+n], basefmt='')
    ax1 = gca()
    for i in range(n, 121+n, 12):
        ax1.annotate('d[%d]'%(i/12), xy=(i-n, d12n[i]), xytext=(i-n+1, d12n[i]))
    xlabel('n (sampling frequency = M$f_i$)')
    ax1.set_ylabel('d12')
    grid('on', ls='dotted')
    ax1.plot([12*5, 12*5], [0, 7000], 'g--')
    ax1.plot([12*4, 12*4], [0, 7000], 'r--')
    ax1.annotate(r'$\alpha=%d$'%n, xy=(12*5, 5500), xytext=(12*4+1, 5500))
    ax1.annotate(r'', xy=(12*4, 5000), xytext=(12*5, 5000), arrowprops=dict(arrowstyle='<->'))

    ax1.plot([12*4+5, 12*4+5], [0, 7000], 'b--')
    ax1.annotate(r'$\delta_1=\alpha-5=%d$'%(n-5), xy=(12*5, 2000), xytext=(12*4+5, 2000))
    ax1.annotate(r'', xy=(12*4+5, 1500), xytext=(12*5, 1500), arrowprops=dict(arrowstyle='<->'))
    ax1.plot([12*4+5+5, 12*4+5+5], [0, 7000], 'b--')
    ax1.annotate(r'$\delta_2=\alpha-10=%d$'%(n-10), xy=(12*5, 3000), xytext=(12*4+5+5, 3000))
    ax1.annotate(r'', xy=(12*4+5+5, 2500), xytext=(12*5, 2500), arrowprops=dict(arrowstyle='<->'))

    ax2 = gca().twinx()
    ax2.plot(flt[7:], 'r')
    ax2.plot(flt[2:], 'r')
    ax2.set_ylim([-0.0195, 0.06])
    plot(np.arange(0, len(flt)-1, 12), flt[7::12], 'gs')
    plot(np.arange(0, len(flt)-1, 12), flt[2::12], 'gs')
    ax2.set_xlim([40, 80])
    save_fig(plt, "image/fsrc_step7.svg")#hide

    clf()
    n = 12
    stem(d12n[n:121+n], basefmt='')
    ax1 = gca()
    for i in range(n, 121+n, 12):
        ax1.annotate('d[%d]'%(i/12), xy=(i-n, d12n[i]), xytext=(i-n+1, d12n[i]))
    xlabel('n (sampling frequency = M$f_i$)')
    ax1.set_ylabel('d12')
    grid('on', ls='dotted')
    ax1.plot([12*5, 12*5], [0, 7000], 'g--')
    ax1.plot([12*4, 12*4], [0, 7000], 'r--')
    ax1.annotate(r'$\alpha=%d$'%n, xy=(12*5, 5500), xytext=(12*4+1, 5500))
    ax1.annotate(r'', xy=(12*4, 5000), xytext=(12*5, 5000), arrowprops=dict(arrowstyle='<->'))

    ax1.plot([12*4+5, 12*4+5], [0, 7000], 'b--')
    ax1.annotate(r'$\delta_1=\alpha-5=%d$'%(n-5), xy=(12*5, 2000), xytext=(12*4+5, 2000))
    ax1.annotate(r'', xy=(12*4+5, 1500), xytext=(12*5, 1500), arrowprops=dict(arrowstyle='<->'))


    ax2 = gca().twinx()
    ax2.plot(np.arange(-7, len(flt)-7, 3), flt[0::3], 'r-s')
    for i in range(0, 121, 3):
        ax2.annotate('f[%d]'%(i/3), xy=(i-7, flt[i]), xytext=(i-7-1, flt[i]+0.003))
    ax2.set_ylim([-0.0195, 0.06])
    ax2.set_xlim([40, 80])
    save_fig(plt, "image/fsrc_step8.svg")#hide

def src_76kto48k():
    #### 1#hide
    # load data
    d = scipy.io.loadmat('./left.mat')
    d = np.squeeze(d['left'])
    fi = 76e3
    fo = 48e3
    # 48kHz/76kHz = 12/19 = M/N
    # so N=19, M=12
    M = 12 #up sampling factor
    N = 19 #down sampling factor

    signal_freqz(d, fi, "image/fsrc_signal.svg")#hide

    #### 2#hide
    # poly-phase filter
    NF = 10 #the filter order in fi (e.g.,76kHz) sampling rate
    flt = scipy.signal.firwin(NF*M+1, 1.0/N)
    delay = (len(flt)-1)/2

    plt.clf()#hide
    plt.subplot(211)#hide
    plt.plot(flt)#hide
    plt.grid('on', ls='dotted')#hide
    plt.subplot(212)#hide
    w, h = scipy.signal.freqz(flt)#hide
    plt.plot(w/np.pi/2*fi*M/1e3, 20*np.log10(np.abs(h)))#hide
    plt.xlabel('f (kHz)')#hide
    plt.ylabel('Magnitude (dB)')#hide
    plt.grid('on', ls='dotted')#hide
    plt.tight_layout()#hide
    save_fig(plt, "image/fsrc_flt.svg")#hide
    plt.show()#noexec#hide

    #### 3#hide
    # the traditional processing, for later comparison
    # up-sampling by a factor of 12
    d12 = scipy.kron(d, np.concatenate((np.ones([1]), np.zeros([M-1]))))
    # filtering
    d12f = scipy.signal.lfilter(flt, 1, d12)
    # down-sampling by a factor of 19, and compensate for the average power loss from upsampling
    d48k = d12f[delay::N]*M
    signal_freqz(d48k, fo, 'image/fsrc_out.svg')#hide

    src_step_by_step(d12[18144:19000])#noexec#hide

    #### 4#hide
    # only saving the right half coefficients, since it is symmetric
    fltp = flt[delay:]
    # append zeros to avoid index overflow
    fltp = np.hstack((fltp, np.zeros((M))))
    # allocate the memory for poly-phase filtering
    d48kp = np.zeros((np.ceil(len(d)/N*M).astype(int)))
    # initialization
    # alpha is the # of samples (@ 76k*M Hz) since last output sample (@76k*M/N Hz)
    # for each input (@76kHz Hz), alpha += M, since @76k*M, it is equivalent to
    # have 1 input sample and M-1 zero samples.
    # if alpha >= N, it means there are at least N samples since the last output.
    # Thus, it needs to output one sample, and alpha-N is the phase. Then alpha
    # needs to be updated as alpha -= N.
    alpha = N-M # so we want to keep the first input sample
    # delay line for input at 76kHz
    bufIn = np.zeros(NF+1)
    for i in range(0, NF/2):
        bufIn[NF/2-i-1] = d[i]

    #### 5#hide
    idx48k = 0
    for i in range(NF/2, len(d)):
        # shift in one data
        bufIn = np.roll(bufIn, 1)
        bufIn[0] = d[i]
        # update the alpha
        alpha = alpha + M
        # need output?
        if alpha >= N:
            delta = alpha - N
            output = 0
            # left side
            for k in range(0, NF/2+1):
                output = output + fltp[delta+k*M]*bufIn[NF/2-k]
            # right side
            delta2 = M-delta
            for k in range(0, NF/2):
                output = output + fltp[delta2+k*M]*bufIn[NF/2+1+k]
            # output
            d48kp[idx48k] = output
            idx48k += 1
            alpha -= N
    #amplitude compensation for up-sampling by a factor of 12
    d48kp = d48kp*M

    #### 6#hide
    plt.clf()
    plt.plot(np.arange(0, len(d48k))/fo*1e3, d48k -d48kp[:len(d48k)])
    plt.xlabel('t (ms)')
    plt.ylabel('diff between classical and poly-phase filtering')
    plt.grid('on', ls='dotted')
    save_fig(plt, "image/fsrc_cmp.svg")#hide
    plt.show()#noexec

