import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal
import scipy.io
def rcosdesign(beta, span, sps, shape='sqrt'):
    """
    Raised cosine FIR pulse-shaping filter design

    return: the filter coefficients whose energy is normalized to be 1
    beta: rolloff factor (0,1)
    span: the single side span (in symbols)
    sps: samples per symbols
    shape: 'sqrt' square-root raised cosine FIR filter
           'normal' raised cosine FIR filter
    """
    assert beta > 0 and beta <= 1 and span > 0 and sps > 0
    N = span*sps
    index = np.linspace(-span, span, N*2+1)
    h = np.zeros((len(index),))
    pi, sin, cos, sqrt = np.pi, np.sin, np.cos, np.sqrt
    if shape == 'sqrt':
        for n, t in enumerate(index):
            if t == 0:
                h[n] = ((1.0-beta)*pi+4.0*beta)/pi
            elif abs((1.0-(4.0*beta*t)**2)) < 1.0e-10:
                h[n] = beta/(pi*sqrt(2.0))*((pi-2.0)*cos(pi/(4.0*beta))+(pi+2.0)*sin(pi/(4.0*beta)))
            else:
                h[n] = sin((1.0-beta)*pi*t) + 4.0*beta*t*cos((1.0+beta)*pi*t)
                h[n] = h[n]/(pi*t*(1.0-(4.0*beta*t)**2))
    elif shape == 'normal':
        for n, t in enumerate(index):
            if abs(1.0-(2.0*beta*t)**2) < 1.0e-10:
                h[n] = pi/4.0*np.sinc(0.5/beta)
            else:
                h[n] = np.sinc(t) * cos(pi*beta*t) / (1.0-(2.0*beta*t)**2)
    return h/sqrt(np.sum(h*h))

def gen_prbs(p, s0, n):
    """
    naive code to generate the pseudo-random binary sequence

    input:
        p: the generator polynomials, for example
            x^7 + x^6 + 1: n.array([0, 0, 0, 0, 0, 1, 1])
        s0: the initial state, same size as p
    return:
        prbs: the prbs array with length n
        s: the current state of the LFSR
    """
    prbs = np.zeros((n,))
    s = s0
    for i in range(n):
        a = np.mod(np.sum(p*s), 2)
        s = np.roll(s, 1)
        s[0] = a
        prbs[i] = a
    return (prbs, s)

def scramble_add(x, p, s0):
    """
    randomize the data with additive scrambler

    input:
        x: input data, 1-dim numpy array
        p: the generator polynomials
        s0: the initial state of the LFSR
    return:
        y: the randomized data (same length as x)
        s: the current state of the LFSR
    """
    # generate the PRBS with same length as x
    prbs, s = gen_prbs(p, s0, x.shape[0])
    # add (mod 2) the PRBS to the data
    y = np.mod(x+prbs, 2)
    return (y, s)

def scramble_add_test():
    #### 1 #hide
    # generate binary data sequence
    x = (np.random.rand(100)+0.5).astype(int)
    # define the generator polynomial
    p = np.array([0, 0, 0, 0, 0, 1, 1])
    # and the initial state of the LSFR
    s0 = np.array([0, 0, 0, 0, 0, 0, 1])
    # randomize the data
    y, s = scramble_add(x, p, s0)
    #### 2 #hide
    np.sum(scramble_add(y, p, s0)[0]-x) #eval

def scramble_mul(x, p, z0=None, mode='normal'):
    """
    randomize the data with multiplicative scrambler

    input:
        x: input data, 1-dim numpy array
        p: the generator polynomial
        z0: the initial buffer status
        mode: 'normal' --> scramble
              'inverse' --> descramble
    return:
        y: the randomized data (same length as x)
        z: the buffer status
    """
    if z0:
        z = z0
    else:
        z = np.zeros(p.shape)
    y = np.zeros(x.shape)
    if mode == 'normal':
        for i in range(x.shape[0]):
            y[i] = np.mod(np.sum(p*z)+x[i], 2)
            z = np.roll(z, 1)
            z[0] = y[i]
    elif mode == 'inverse':
        for i in range(x.shape[0]):
            y[i] = np.mod(np.sum(p*z)+x[i], 2)
            z = np.roll(z, 1)
            z[0] = x[i]
    return (y, z)

def scramble_mul_test():
    #### 1#hide
    # generate binary data sequence
    x = (np.random.rand(100)+0.5).astype(int)
    # define the generator polynomial
    p = np.array([0]*13 + [1] + [0]*2 + [1])
    y, s = scramble_mul(x, p)
    np.sum(scramble_mul(y, p, mode='inverse')[0]-x)#eval

def interleaver_block(x, n):
    """
    block interleaver

    input:
        x: 1-dim numpy array
        n: the dimension of the interleaver, the length of x should be integral multiples of n
    return:
        y: the interleaved data
    """
    y = np.reshape(x, (n, -1)).T.flatten()
    return y

def interleaver_block_test():
    #### 1#hide
    x = np.arange(1000)
    # interleaver
    y = interleaver_block(x, 20)
    # deinterleaver
    xd = interleaver_block(y, 50)
    np.allclose(x, xd)#eval

def interleaver_conv(x, nrows, slope, state=None, mode='normal'):
    """
    convolutional interleaver

    input:
        x: 1-dim numpy arry
        nrows: the number of rows
        slope: the ith row will have i*slope registers
        state: a dict
            'value': the initial value of the internal registers
            'index': the initial index
        mode:
            'normal': interleaver
            'deintlv': de-interleaver
    """
    if state == None:
        # the only difference between the interleaver and de-interleaver is the
        # order of the shift registers.
        if mode == 'normal':
            value = [np.zeros(n*slope) for n in range(nrows)]
        elif mode == 'deintlv':
            value = [np.zeros(n*slope) for n in reversed(list(range(nrows)))]
        index = 0
    else:
        value = state['value']
        index = state['index']
    y = np.zeros(x.shape)
    for r in range(nrows):
        # process the data in each row
        n = np.mod(r + index, nrows)
        xn = np.hstack([value[n], x[n::nrows]])
        y[n::nrows] = xn[:y[n::nrows].shape[0]]
        value[n] = xn[y[n::nrows].shape[0]:]
    index = np.mod(x.shape[0] + index, nrows)
    state = {'value': value, 'index':index}
    return (y, state)

def interleaver_conv_test():
    #### 1#hide
    x = np.arange(1000)
    # interleaver
    y, istate = interleaver_conv(x, 3, 4)
    # de-interleaver
    xd, dstate = interleaver_conv(y, 3, 4, mode='deintlv')
    # calculate the delay from the interleaver and de-interleaver, which is
    # equal to the total number of the shift registers
    delay = int(2*4*(0+3-1)*3/2)
    np.allclose(x[:-delay], xd[delay:])#eval

def bcd2gray(b):
    """
    get the corresponding Gray from the BCD code

    input
        b: BCD code

    return: its Gray BCD
    """
    c = np.roll(b, 1)
    c[0] = 0
    return np.logical_xor(b, c).astype(int)

def gray2bcd(G):
    """
    get the corresponding BCD from the Gray code

    input
        G: Gray code

    return: its corresponding BCD
    """
    b = np.copy(G)
    for i in range(1, b.shape[0]):
        b[i] = np.logical_xor(b[i-1], G[i])
    return b.astype(int)

def gen_gray_table(n):
    """
    generate 2-d Gray constellation table

    input
        n: the number of bits
    return
        g: the mapping from Gray code (source bits) to BCD (symbol (x, y) on the
           constellation), where ith row is for BCD(i)
    """
    # generate the binary representation of BCD
    power_of_two = 2**np.arange(n-1,-1,-1)
    t = (np.arange(2**n)[:, np.newaxis] & power_of_two) / power_of_two

    # split into two groups
    g1 = t[:, 0:n/2]
    g2 = t[:, n/2:]
    g1 = np.apply_along_axis(gray2bcd, axis=1, arr=g1)
    g2 = np.apply_along_axis(gray2bcd, axis=1, arr=g2)
    g1 = np.dot(g1, 2**np.arange(n/2-1, -1, -1))
    g2 = np.dot(g2, 2**np.arange(n-n/2-1, -1, -1))
    g = np.stack([g1, g2]).T
    return g

def groupdelay_test():
    from scipy import signal, pi
    b = list(range(10, 0, -1))
    w, gd = signal.group_delay((b, 1))
    import matplotlib.pyplot as plt
    plt.plot(w/pi/2, gd)
    plt.ylabel('Group delay (samples)')
    plt.xlabel('Normalized frequency')
    plt.grid('on', ls='dotted')
    plt.xlim([0, 0.5])
    plt.show()

    ####
    plt.plot(w/pi/2, gd)
    plt.ylabel('Group delay (samples)')
    plt.xlabel('Normalized frequency')
    plt.grid('on', ls='dotted')
    plt.xlim([0, 0.5])
    plt.savefig("image/groupdelay.svg")

def halfband_test():
    from scipy import signal, pi
    b = list(range(10, 0, -1))
    w, gd = signal.group_delay((b, 1))
    import matplotlib.pyplot as plt
    plt.plot(w/pi/2, gd)
    plt.ylabel('Group delay (samples)')
    plt.xlabel('Normalized frequency')
    plt.grid('on', ls='dotted')
    plt.xlim([0, 0.5])
    plt.show()

    ####
    plt.plot(w/pi/2, gd)
    plt.ylabel('Group delay (samples)')
    plt.xlabel('Normalized frequency')
    plt.grid('on', ls='dotted')
    plt.xlim([0, 0.5])
    plt.savefig("image/groupdelay.svg")

def fir_test_float():
    # normalized signal frequency
    fs = 0.01
    s = np.sin(2*np.pi*fs*np.arange(1000))
    # normalized noise frequency
    fn = 0.2
    n = np.sin(2*np.pi*fn*np.arange(1000))
    plt.plot(n+s)
    plt.show()
    # estimate the filter length, 60dB suppression
    N = np.ceil(60/22/((fn-fs))).astype(int)
    b = scipy.signal.firwin(N, (fn+fs)/2)
    y = scipy.signal.lfilter(b, 1, s+n)
    plt.plot(s)
    plt.plot(y, 'r')
    plt.show()

def save_fig(plt, filename):
    import os
    if not os.path.isfile(filename):
        plt.savefig(filename)

def fir_test_fixed():
    #### 1 #hide
    # normalized signal frequency
    fs = 0.02
    s = np.sin(2*np.pi*fs*np.arange(1000))
    # normalized noise frequency
    fn = 0.2
    n = np.sin(2*np.pi*fn*np.arange(1000))
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(s)
    plt.plot(n)
    plt.legend(['signal', 'noise'])
    plt.xlim([0, 100])
    plt.subplot(2, 1, 2)
    plt.plot(np.linspace(0, 1, len(s), endpoint=False), 20*np.log10(abs(np.fft.fft(s+n))))
    plt.xlabel('Normalized frequency')
    save_fig(plt, "image/fir_demo_signal.svg")#hide
    plt.show()#noexec

    #### 2 #hide
    # normalize the input signal to 8 bits (3 bits before the decimal point)
    plt.clf()#hide
    r = np.round((s+n)*2**5)/2**5
    plt.plot(r-n-s, 'r')
    plt.title('difference between the float and fixed input')
    save_fig(plt, "image/fir_demo_signal_fixed.svg")#hide
    plt.show()#noexec
    #### 3 #hide
    # estimate the number of filter coefficients
    N = np.ceil(60/22/((fn-fs))).astype(int)
    # design the low-pass filter
    b = scipy.signal.firwin(N, (fn+fs)/2)
    b #eval
    # apply the filter to input
    y = scipy.signal.lfilter(b, 1, r)
    plt.clf()#hide
    plt.plot(r, 'b')
    plt.plot(s, 'g')
    plt.plot(y, 'r')
    plt.xlim([0, 100])
    plt.legend(['signal+noise', 'signal', 'filtered'])
    save_fig(plt, "image/fir_demo_signal_filtered.svg")#hide
    plt.show()#noexec

    #### 4 #hide
    Nc = 8
    bf = np.round(b*2**(Nc+2))/2**(Nc+2)
    yf = scipy.signal.lfilter(bf, 1, r)
    plt.clf()#hide
    plt.plot(y-yf)
    save_fig(plt, "image/fir_demo_signal_filtered_fix.svg")#hide
    plt.show()#noexec

    #### 5 #hide
    mse = []
    NC = np.arange(4, 20)
    for Nc in NC:
        bf = np.round(b*2**(Nc+2))/2**(Nc+2)
        yf = scipy.signal.lfilter(bf, 1, r)
        mse.append(np.sqrt(np.mean((yf-y)**2)))
    plt.clf()#hide
    plt.semilogy(NC, mse, '-o')
    plt.grid('on', ls='dotted')
    plt.ylabel('MSE')
    plt.xlabel('# of effective bits of filter coefficients')
    save_fig(plt, "image/fir_demo_signal_filtered_fix_mse.svg")#hide
    plt.show()#noexec

    #### 6 #hide
    mse_f = []
    for Nc in NC:
        bf = np.floor(b*2**(Nc+2))/2**(Nc+2)
        yf = scipy.signal.lfilter(bf, 1, r)
        mse_f.append(np.sqrt(np.mean((yf-y)**2)))
    plt.clf()#hide
    plt.semilogy(NC, mse, '-o')
    plt.semilogy(NC, mse_f, 'r-s')
    plt.grid('on', ls='dotted')
    plt.ylabel('MSE')
    plt.xlabel('# of effective bits of filter coefficients')
    plt.legend(['round', 'floor'])
    save_fig(plt, "image/fir_demo_signal_filtered_fix_mse2.svg")#hide
    plt.show()#noexec

    #### 7 #hide
    Nc = 8
    bf = np.round(b*2**(Nc+2))/2**(Nc+2)
    yf = scipy.signal.lfilter(bf, 1, r)
    plt.clf()#hide
    mse_y = []
    for Ny in range(3, 20):
        yf_t = np.round(yf*2**Ny)
        # saturation
        yf_t[yf_t >= 2**(Ny+3-1)] = 2**(Ny+3-1)-1
        yf_t[yf_t < -2**(Ny+3-1)] = -2**(Ny+3-1)
        yf_t = yf_t/2**Ny
        mse_y.append(np.sqrt(np.mean((yf_t-y)**2)))
    plt.semilogy(np.arange(3, 20)+3, mse_y, '-o')
    plt.grid('on', ls='dotted')
    plt.ylabel('MSE')
    plt.xlabel('# of effective output bits')
    save_fig(plt, "image/fir_demo_signal_filtered_fix_msey.svg")#hide
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

def _cordic(a, b, iteration=10):
    """
    return the phase of complex value a+jb

    the phase must be in [0, pi/4]
    """
    alpha = 0
    d = -1
    atan = np.arctan(1./np.power(2., np.arange(1, iteration+1)))
    for k in range(iteration):
        a_n = a - b*d*2**(-k-1)
        b_n = b + a*d*2**(-k-1)
        alpha = alpha - d * atan[k]
        if b_n < 0: d = 1
        else: d = -1
        a, b = a_n, b_n
    return alpha

def cordic(a, b, iteration=10):
    """
    return the phase of complex value a+jb

    return the phase (-pi, pi]
    """
    a_m = np.max((np.abs(a), np.abs(b)))
    b_m = np.min((np.abs(a), np.abs(b)))
    alpha = _cordic(a_m, b_m, iteration)
    if a_m != np.abs(a):
        alpha = np.pi/2-alpha
    if a < 0 and b >= 0:
        alpha = np.pi - alpha
    elif a < 0 and b < 0:
        alpha = alpha - np.pi
    elif a >= 0 and b < 0:
        alpha = -alpha
    return alpha

def cordic_test():
    alpha = np.linspace(-np.pi, np.pi, 10000)
    alpha_err = np.zeros(10)
    for iteration in range(5, 15):
        t = np.zeros_like(alpha)
        for i, p in enumerate(alpha):
            t[i] = cordic(np.cos(p), np.sin(p), iteration)
        alpha_err[iteration - 5] = np.max(np.abs(t-alpha))
    plt.clf()
    plt.plot(np.arange(5, 15), alpha_err*180/np.pi, '-o')
    plt.xlabel('iterations')
    plt.ylabel('max estimation error (in degree)')
    plt.grid('on', ls=':')
    save_fig(plt, '../doc/image/sync_cordic_est.svg')#hide
