import numpy as np
import matplotlib.pyplot as plt
import common

def basics_detection_binary():
    x = np.linspace(-4, 4, 1000)
    plt.clf()
    plt.plot(x, 1/np.sqrt(2*np.pi)*np.exp(-(x-1)**2/2))
    plt.plot(x, 1/np.sqrt(2*np.pi)*np.exp(-(x+1)**2/2))
    plt.plot([0, 0], [0, 0.5], '-.')
    plt.grid('on', ls=":")
    plt.xlabel(r'$x$')
    plt.ylabel(r'$p(x|s)$')
    plt.legend([r'$s=1$', r'$s={-1}$'])
    save_fig(plt, '../doc/image/basics_det_ml_prop.png')

    plt.clf()
    plt.plot(x, 0.3*1/np.sqrt(2*np.pi)*np.exp(-(x-1)**2/2))
    plt.plot(x, 0.7*1/np.sqrt(2*np.pi)*np.exp(-(x+1)**2/2))
    plt.plot(np.log(0.7/0.3)/2*np.array([1, 1]), [0, 0.3], '-.')
    plt.grid('on', ls=":")
    plt.xlabel(r'$x$')
    plt.ylabel(r'$p(x|s)p(s)$')
    plt.legend([r'$s=1$', r'$s={-1}$'])
    save_fig(plt, '../doc/image/basics_det_map_prop.png')

    plt.clf()
    plt.plot(x, 0.7*1/np.sqrt(2*np.pi)*np.exp(-(x-1)**2/2))
    plt.plot(x, 0.3*1/np.sqrt(2*np.pi)*np.exp(-(x+1)**2/2))
    plt.plot(np.log(0.3/0.7)/2*np.array([1, 1]), [0, 0.3], '-.')
    plt.grid('on', ls=":")
    plt.xlabel(r'$x$')
    plt.ylabel(r'$p(x|s)p(s)$')
    plt.legend([r'$s=1$', r'$s={-1}$'])
    save_fig(plt, '../doc/image/basics_det_map_prop2.png')
