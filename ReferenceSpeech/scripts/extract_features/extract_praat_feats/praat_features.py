""" 
This script computes features using praat software, and the parselmouth 
package to use praat with python.

Particularly, it computes:
- hnr, jitter, shimmer based features
- formant based features
- speech rythm features

This is strongly based on https://osf.io/umrjq/ and
https://osf.io/r8jau/?ref=9cf51e2b08b5eda9a721a945ae5690ce655e3276 
""" 
 
import statistics
import math
import numpy as np

import parselmouth
from parselmouth.praat import call

def praat_audio_reader(path):
    # reads audio from path
    return parselmouth.Sound(path)

def pitch_hnr_jitter_shimmer(sound, f0min=75, f0max=600, unit="Hertz"):
    """
    adapted from https://osf.io/umrjq/ 

    This is the function to measure source acoustics 
    using default male parameters.

    The values in queries here refer to praat defaults.
    """

    duration = call(sound, "Get total duration") # duration
    
    # pitch
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
    meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
    stdevF0 = call(pitch, "Get standard deviation", 0 ,0, unit) # get standard deviation
    minF0 = call(pitch, "Get minimum", 0, 0, unit, "parabolic") # get min pitch
    maxF0 = call(pitch, "Get maximum", 0, 0, unit, "parabolic") # get max pitch


    # harmonics-to-noise Ratio
    # agrs: time step (s); minimum_picth (Hz); Silence threshold; periods per window
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    
    # jitter
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    # args: time_range_in; time_range_fin (==0 -> all); shortest period; longest period; maximum_period
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    
    # shimmer
    # args: time_range_in; time_range_fin (==0 -> all); shortest period; longest period; maximum_period; maximum amplitude factor
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    
    f0_hnr_jit_shi_dict = {
        'duration(s)': duration,
        'meanF0': meanF0,
        'stdevF0':stdevF0,
        'minF0':minF0,
        'maxF0':maxF0,
        'hnr':hnr,
        'localJitter':localJitter, 
        'localabsoluteJitter':localabsoluteJitter, 
        'rapJitter':rapJitter, 
        'ppq5Jitter':ppq5Jitter, 
        'ddpJitter': ddpJitter, 
        'localShimmer':localShimmer, 
        'localdbShimmer':localdbShimmer, 
        'apq3Shimmer':apq3Shimmer, 
        'aqpq5Shimmer':aqpq5Shimmer, 
        'apq11Shimmer':apq11Shimmer, 
        'ddaShimmer':ddaShimmer
        }
    
    return f0_hnr_jit_shi_dict

def measureFormants(sound, f0min=50, f0max=500, unit="Hertz"):
    """
    from https://osf.io/umrjq/
    This function measures formants using Formant Position formula
    """
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    
    # args: time step (s); max number of formants; formant ceiling (Hz); window length (s); pre-emphasis from (Hz)
    formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
    numPoints = call(pointProcess, "Get number of points")

    f1_list = []
    f2_list = []
    f3_list = []
    f4_list = []
    
    # Measure formants only at glottal pulses
    for point in range(0, numPoints):
        point += 1
        t = call(pointProcess, "Get time from index", point)

        # args: formant number; time (s); unit; interpolation
        f1 = call(formants, "Get value at time", 1, t, unit, 'Linear')
        f2 = call(formants, "Get value at time", 2, t, unit, 'Linear')
        f3 = call(formants, "Get value at time", 3, t, unit, 'Linear')
        f4 = call(formants, "Get value at time", 4, t, unit, 'Linear')
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)
        f4_list.append(f4)
    
    f1_list = [f1 for f1 in f1_list if str(f1) != 'nan']
    f2_list = [f2 for f2 in f2_list if str(f2) != 'nan']
    f3_list = [f3 for f3 in f3_list if str(f3) != 'nan']
    f4_list = [f4 for f4 in f4_list if str(f4) != 'nan']
    
    # calculate mean formants across pulses
    f1_mean = statistics.mean(f1_list)
    f2_mean = statistics.mean(f2_list)
    f3_mean = statistics.mean(f3_list)
    f4_mean = statistics.mean(f4_list)
    
    # calculate median formants across pulses
    f1_median = statistics.median(f1_list)
    f2_median = statistics.median(f2_list)
    f3_median = statistics.median(f3_list)
    f4_median = statistics.median(f4_list)
    
    formant_dict = {
        'f1_mean': f1_mean, 
        'f2_mean': f2_mean, 
        'f3_mean': f3_mean, 
        'f4_mean': f4_mean, 
        'f1_median':f1_median, 
        'f2_median':f2_median, 
        'f3_median': f3_median, 
        'f4_median': f4_median
    }
    return formant_dict


def speech_rate(sound, silencedb=-25, mindip=2, minpause=0.3):
    """
    from: https://osf.io/r8jau/?ref=9cf51e2b08b5eda9a721a945ae5690ce655e3276
    First and last silences from each audio (before and after voice) 
    are not included in the comptuation of speech rate features.
    """
    originaldur = sound.get_total_duration()
    intensity = sound.to_intensity(50)
    start = call(intensity, "Get time from frame number", 1)
    nframes = call(intensity, "Get number of frames")
    end = call(intensity, "Get time from frame number", nframes)
    min_intensity = call(intensity, "Get minimum", 0, 0, "Parabolic")
    max_intensity = call(intensity, "Get maximum", 0, 0, "Parabolic")

    # get .99 quantile to get maximum (without influence of non-speech sound bursts)
    max_99_intensity = call(intensity, "Get quantile", 0, 0, 0.99)

    # estimate Intensity threshold
    threshold = max_99_intensity + silencedb
    threshold2 = max_intensity - max_99_intensity
    threshold3 = silencedb - threshold2
    if threshold < min_intensity:
        threshold = min_intensity

    # get pauses (silences) and speakingtime
    textgrid = call(intensity, "To TextGrid (silences)", threshold3, minpause, 0.1, "silent", "sounding")
    silencetier = call(textgrid, "Extract tier", 1)
    silencetable_sounds = call(silencetier, "Down to TableOfReal", "sounding")
    n_speech_segs = call(silencetable_sounds, "Get number of rows")
    if n_speech_segs > 1:
        silencetable_sil = call(silencetier, "Down to TableOfReal", "silent")
        n_sil_segs = call(silencetable_sil, "Get number of rows")
    else:
        n_sil_segs = 0
    
    # speaking duration
    speakingtot = 0
    speech_segs_dur = []
    for ispeech in range(1, n_speech_segs+1):
        beginsound = call(silencetable_sounds, "Get value", ispeech, 1)
        endsound = call(silencetable_sounds, "Get value", ispeech, 2)
        speakingdur = endsound - beginsound
        speakingtot += speakingdur
        speech_segs_dur.append(speakingdur)

    # pauses duration
    silenttot = 0
    sil_pauses_dur = []
    first_sil_dur = 0    # 0 in case there is no initial silence
    last_sil_dur = 0     # 0 in case there is no final silence
    for ipause in range(1, n_sil_segs+1):
        beginsil = call(silencetable_sil, "Get value", ipause, 1)
        endsil = call(silencetable_sil, "Get value", ipause, 2)
        if beginsil == 0: # excludes first silence if it occurs before speech 
            first_sil_dur = endsil - beginsil
            continue            
        if endsil > endsound: # excludes last silence if it occurs after speech 
            last_sil_dur = endsil - beginsil
            continue
        sildur = endsil - beginsil
        silenttot += sildur
        sil_pauses_dur.append(sildur)

    # original dur except first and last silences, 
    # if they occur before and after first speech
    dur_between_speech = originaldur - first_sil_dur - last_sil_dur

    intensity_matrix = call(intensity, "Down to Matrix")
    # sndintid = sound_from_intensity_matrix
    sound_from_intensity_matrix = call(intensity_matrix, "To Sound (slice)", 1)
    # use total duration, not end time, to find out duration of intdur (intensity_duration)
    # in order to allow nonzero starting times.
    intensity_duration = call(sound_from_intensity_matrix, "Get total duration")
    intensity_max = call(sound_from_intensity_matrix, "Get maximum", 0, 0, "Parabolic")
    point_process = call(sound_from_intensity_matrix, "To PointProcess (extrema)", "Left", "yes", "no", "Sinc70")
    # estimate peak positions (all peaks)
    numpeaks = call(point_process, "Get number of points")
    t = [call(point_process, "Get time from index", i + 1) for i in range(numpeaks)]

    # fill array with intensity values
    timepeaks = []
    peakcount = 0
    intensities = []
    for i in range(numpeaks):
        value = call(sound_from_intensity_matrix, "Get value at time", t[i], "Cubic")
        if value > threshold:
            peakcount += 1
            intensities.append(value)
            timepeaks.append(t[i])

    # fill array with valid peaks: only intensity values if preceding
    # dip in intensity is greater than mindip
    validpeakcount = 0
    currenttime = timepeaks[0]
    currentint = intensities[0]
    validtime = []

    for p in range(peakcount - 1):
        following = p + 1
        followingtime = timepeaks[p + 1]
        dip = call(intensity, "Get minimum", currenttime, timepeaks[p + 1], "None")
        diffint = abs(currentint - dip)
        if diffint > mindip:
            validpeakcount += 1
            validtime.append(timepeaks[p])
        currenttime = timepeaks[following]
        currentint = call(intensity, "Get value at time", timepeaks[following], "Cubic")

    # Look for only voiced parts
    pitch = sound.to_pitch_ac(0.02, 30, 4, False, 0.03, 0.25, 0.01, 0.35, 0.25, 450)
    voicedcount = 0
    voicedpeak = []

    for time in range(validpeakcount):
        querytime = validtime[time]
        whichinterval = call(textgrid, "Get interval at time", 1, querytime)
        whichlabel = call(textgrid, "Get label of interval", 1, whichinterval)
        value = pitch.get_value_at_time(querytime) 
        if not math.isnan(value):
            if whichlabel == "sounding":
                voicedcount += 1
                voicedpeak.append(validtime[time])

    # calculate time correction due to shift in time for Sound object versus
    # intensity object
    timecorrection = originaldur / intensity_duration

    # Insert voiced peaks in TextGrid
    call(textgrid, "Insert point tier", 1, "syllables")
    for i in range(len(voicedpeak)):
        position = (voicedpeak[i] * timecorrection)
        call(textgrid, "Insert point", 1, position, "")

    # return results
    speakingrate = voicedcount / dur_between_speech
    articulationrate = voicedcount / speakingtot
    npause = n_speech_segs - 1 # considers only pauses bounded by speech segs
    if voicedcount:
        asd = speakingtot / voicedcount
    else:
        print ("[WARNING]: NO VOICED SEGMENTS IDENTIFIED")
        asd = None


    # silence feats (inspired in Weiner et al. 2016)
    mean_pause_dur = np.mean(sil_pauses_dur) if len(sil_pauses_dur) else 0 # mean dur of silence segm
    mean_speech_dur = np.mean(speech_segs_dur) if len(speech_segs_dur) else 0 # mean dur of speech segms
    silence_rate = silenttot / dur_between_speech # total silence duration / total duration between speech
    silence_speech_ratio = npause/n_speech_segs # n silent segmnets / n speech segments
    mean_sil_count = npause / dur_between_speech # n silence segments / total duration between speech

    longsils = [s for s in sil_pauses_dur if s>1]
    lsil_rate = np.sum(longsils) / dur_between_speech 
    lsil_speech_ratio = len(longsils) / n_speech_segs
    mean_lsil_count = len(longsils) / dur_between_speech 

    speechrate_dictionary = {
                            'nsyll': voicedcount,
                            'npause': npause,
                            'dur(s)': originaldur,
                            'phonationtime(s)': speakingtot,
                            'silencetime(s)':silenttot,
                            'speechrate(nsyll / dur)': speakingrate,
                            "articulation rate(nsyll / phonationtime)": articulationrate,
                            "ASD(speakingtime / nsyll)": asd,
                            "mean_pause_dur": mean_pause_dur,
                            "mean_speech_dur": mean_speech_dur,
                            "silence_rate (silencetime/dur)": silence_rate,
                            "silence_speech_ratio": silence_speech_ratio,
                            "mean_sil_count": mean_sil_count,
                            "lsil_rate": lsil_rate,
                            "lsil_speech_ratio": lsil_speech_ratio,
                            "mean_lsil_count": mean_lsil_count}
    return speechrate_dictionary
