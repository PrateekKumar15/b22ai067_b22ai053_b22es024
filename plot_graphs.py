import matplotlib.pyplot as plt
import numpy as np
import qrcode
import os

def generate_spectrograms():
    """Generates the spectrogram.png comparing clean vs noisy audio"""
    print("Generating spectrograms...")
    fs = 16000
    t = np.linspace(0, 1, fs)
    # clean signal
    clean = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
    # noisy signal
    noise = np.random.normal(0, 0.5, fs)
    noisy = clean + noise

    plt.figure(figsize=(10, 4))
    
    # Subplot 1
    plt.subplot(1, 2, 1)
    plt.specgram(clean, Fs=fs, cmap='magma')
    plt.title('Clean Acoustic Features')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    
    # Subplot 2
    plt.subplot(1, 2, 2)
    plt.specgram(noisy, Fs=fs, cmap='magma')
    plt.title('Acoustic Degradation (Low SNR)')
    plt.xlabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig('spectrogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved spectrogram.png")

def generate_qrcode():
    """Generates a QR code image to place into the poster"""
    print("Generating QR code...")
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data("https://github.com/PrateekKumar15/Speech_Project.git")
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save('qrcode.png')
    print("Saved qrcode.png")

if __name__ == "__main__":
    generate_spectrograms()
    generate_qrcode()
    print("All assets generated successfully in the root directory!")
