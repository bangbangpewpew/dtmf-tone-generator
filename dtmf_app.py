import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.fft import fft
import wave

# DTMF frequency table
dtmf_freqs = {
    '1': (697, 1209),
    '2': (697, 1336),
    '3': (697, 1477),
    '4': (770, 1209),
    '5': (770, 1336),
    '6': (770, 1477),
    '7': (852, 1209),
    '8': (852, 1336),
    '9': (852, 1477),
    '0': (941, 1336),
    '*': (941, 1209),
    '#': (941, 1477)
}

def generate_dtmf_tone(key, duration=0.5, sampling_rate=8000):
    """Generate a DTMF tone for a given key."""
    if key not in dtmf_freqs:
        raise ValueError("Invalid DTMF key.")
    
    # Get corresponding frequencies
    low_freq, high_freq = dtmf_freqs[key]

    # Create time vector
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    # Generate sine waves
    low_tone = np.sin(2 * np.pi * low_freq * t)
    high_tone = np.sin(2 * np.pi * high_freq * t)

    # Create final DTMF tone
    dtmf_tone = low_tone + high_tone
    return dtmf_tone

def save_wav_file(signal, filename, samplerate=8000):
    """Save the generated DTMF tone as a WAV file."""
    # Normalize to 16-bit range
    scaled_signal = np.int16(signal / np.max(np.abs(signal)) * 32767)

    # Write to WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16 bits
        wf.setframerate(samplerate)
        wf.writeframes(scaled_signal.tobytes())

def plot_time_domain(signal, time):
    """Plot the time-domain signal."""
    plt.figure(figsize=(10, 4))
    plt.plot(time[:100], signal[:100])
    plt.title("Time Domain Signal")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid()
    st.pyplot(plt)

def plot_frequency_spectrum(signal, sampling_rate=8000):
    """Plot the frequency spectrum using FFT."""
    n = len(signal)
    freq = np.fft.fftfreq(n, 1/sampling_rate)
    spectrum = np.abs(fft(signal))

    plt.figure(figsize=(10, 4))
    plt.plot(freq[:n // 2], spectrum[:n // 2])
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()
    st.pyplot(plt)

def identify_key(frequencies, magnitudes, tolerance=20):
    """Identify the DTMF key based on the frequency spectrum."""
    peaks_indices = np.argsort(magnitudes)[-2:]  # Get indices of two largest peaks
    detected_freqs = frequencies[peaks_indices]
    detected_freqs.sort()  # Sort to match DTMF structure

    for key, (low, high) in dtmf_freqs.items():
        # Check if the detected frequencies are within the tolerance range
        if (any(abs(low - freq) < tolerance for freq in detected_freqs) and
                any(abs(high - freq) < tolerance for freq in detected_freqs)):
            return key
    return "Unknown Key"

# Streamlit UI
st.title("DTMF Tone Generator and Analyzer")

# User input for multiple DTMF keys
keys = st.text_input("Enter DTMF keys (e.g., 123*#):", value="123")

duration = st.slider("Select duration per tone (seconds):", 0.1, 1.0, 0.5)

if st.button("Generate DTMF Tone"):
    total_tone = np.array([])  # Empty array to hold the final concatenated signal
    for key in keys:
        if key in dtmf_freqs:
            dtmf_tone = generate_dtmf_tone(key, duration)
            total_tone = np.concatenate([total_tone, dtmf_tone])
        else:
            st.write(f"Invalid key: {key}. Ignoring.")
    
    if len(total_tone) > 0:
        # Save the generated tone as a WAV file
        wav_filename = "dtmf_tone.wav"
        save_wav_file(total_tone, wav_filename)

        # Use Streamlit's audio function to play the sound
        st.audio(wav_filename)

        # Create a time vector for plotting
        time_vector = np.linspace(0, len(total_tone) / 8000, len(total_tone), endpoint=False)
        
        # Plot time-domain signal
        plot_time_domain(total_tone, time_vector)

        # Plot frequency spectrum
        plot_frequency_spectrum(total_tone)

        # Apply DFT and identify key
        n = len(total_tone)
        freq = np.fft.fftfreq(n, 1/8000)
        spectrum = np.abs(fft(total_tone))
        detected_key = identify_key(freq[:n // 2], spectrum[:n // 2])

        st.write(f"Detected DTMF Key: {detected_key}")
    else:
        st.write("No valid keys entered.")
