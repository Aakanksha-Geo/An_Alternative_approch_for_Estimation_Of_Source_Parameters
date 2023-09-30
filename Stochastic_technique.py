import numpy as np
import matplotlib.pyplot as plt
from numpy import random

t_final = 25
dt = 0.005
np.random.seed(1)
noise = np.random.normal(0.0, 1.1, (int(t_final/dt,)))
t = np.arange(0, t_final, dt)

# plt.plot(t, noise, color='black')
# plt.ylim(-5.5, 5.9)
# plt.ylabel("Noise", fontsize=16)
# plt.xlabel("Time", fontsize=16)
# plt.title("Random Noise", fontsize=16)
# # plt.show()


def Window(t):
    eps = 0.05001000000000001
    eta = 1e-09
    T = t[-1]
    t_eta = 3*T
    b = -(eps*np.log(eta)/(1 + eps*(np.log(eps)-1)))
    a = (np.exp(1)/eps)**b
    c = b/eps
    w = a*(t/t_eta)**b * np.exp(-c*(t/t_eta))
    return w


W = Window(t)
plt.plot(t, W, color='black')
plt.xlabel("Time", fontsize=16)
plt.title("Window Function", fontsize=16)

win_noise = noise*W
print(win_noise)

# plt.plot(t, win_noise, color='black')
# plt.xlabel("Time(sec)", fontsize=16)
# plt.title("Windowed Noise", fontsize=16)


def Np_fft(y, t, dt):
    fft = np.fft.rfft(y, norm='ortho')
    # N = len(t)
    N = t.shape[0]
    freq = np.fft.rfftfreq(N, dt)
    print(freq, fft, len(freq), len(fft))
    return freq[1:], fft[1:], abs(fft)[1:]


freq, noise_spectrum_complex, noise_spectrum = Np_fft(win_noise, t, dt)

# plt.plot(freq, noise_spectrum, color='black', lw=1)
# plt.xscale("log")
# plt.yscale("log")
# # plt.xlim(0.001,np.log(50))
# plt.ylabel("Fourier Amplitde", fontsize=16)
# plt.xlabel("Frequency", fontsize=16)
# plt.title("Noise Spectrum", fontsize=16)


# N = 1024*10
# dt = 0.005
# I = np.arange(1, (N/2)+1, 1)
# fi = I/(N*dt)
fi = freq


def a(f, R):
    M0 = 3.2e23  # dyne - cm
    Rho = 2.7   # g/cm**3
    V = 3.2e5  # cm/sec
    k = 0.03
    # Q = 180*f**0.45
    Q = 110*f**1.02
    fc = 1.07
    Rtp = 0.60
    FS = 2
    PR = 1/np.sqrt(2)
    C = (Rtp*FS*PR)/(4*np.pi*Rho*V**3)
    ans = (C*M0*((2*np.pi*f)**2)*np.exp((-np.pi*f*R)/(Q*V))
           * np.exp(-np.pi*k*f))/((1+(f/fc)**2)*R)
#     print(ans)
    return ans


Amplitude_spectrum = a(fi, 25e5)
# plt.figure(figsize=(7, 5))
# plt.xscale("log")
# plt.yscale("log")
# # plt.xlim(0,5)
# plt.plot(fi, Amplitude_spectrum, lw=1, c="black")
# plt.xlabel("Frequency", fontsize=16)
# plt.ylabel("Amplitude", fontsize=16)
# plt.title("Amplitude Spectrum", fontsize=16)
# # plt.grid()
# plt.tight_layout()
# # plt.savefig('Model_Spectrum.png', format='png', dpi=350)
# # plt.show()


shaped_noise_spectra = Amplitude_spectrum*noise_spectrum
shaped_noise_spectra_complex = Amplitude_spectrum*noise_spectrum_complex

plt.plot(fi, shaped_noise_spectra, color='black', lw=1)
plt.xscale("log")
plt.yscale("log")
# plt.xlim(0.001, np.log(50))
plt.ylabel("Fourier Amplitde", fontsize=16)
plt.xlabel("Frequency", fontsize=16)
plt.title("Shaped noise", fontsize=16)
plt.show()


def Np_ifft(y, n):
    isp = 2*np.fft.irfft(y, n, norm='forward')
    print(isp.shape)
    return isp.real, isp


accr, acc = Np_ifft(shaped_noise_spectra_complex, len(win_noise))

# plt.plot(t, accr, color='black')
# plt.xlabel("Time(sec)", fontsize=16)
# plt.ylabel(r"Acceleration (cm/$s^2$)", fontsize=16)
# plt.title("Accelerogram", fontsize=16)

freq, acc_spectrum_complex, acc_spectrum = Np_fft(accr, t, dt)
acc_spectrum = acc_spectrum*dt

plt.plot(fi, acc_spectrum, color='black', lw=1)
plt.plot(fi, shaped_noise_spectra, color='red', lw=1)
plt.xscale("log")
plt.yscale("log")
plt.ylim(1.5*10**(-5), )
plt.ylabel("Fourier Amplitde", fontsize=16)
plt.xlabel("Frequency", fontsize=16)
plt.show()


