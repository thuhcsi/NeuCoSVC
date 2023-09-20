<!-- # Neural Concatenative Singing Voice Conversion -->

## Abstract

Any-to-any singing voice conversion is confronted with a significant challenge of "timbre leakage" issue caused by insufficient disentanglement between the content and the speaker timbre. To address the formidable challenge of disentanglement, this study introduces a novel neural concatenative singing voice conversion (NeuCoSVC) framework. The NeuCoSVC framework comprises a self-supervised learning (SSL) representation extractor, a neural harmonic signal generator, and a waveform synthesizer. 
The SSL model condenses the audio into a sequence of fixed-dimensional SSL features. The harmonic signal generator produces both raw and filtered harmonic signals leveraging a linear time-varying filter given condition features. Simultaneously, the audio generator creates waveforms directly from the SSL features, integrating both the harmonic signals and the loudness. During inference, the audio generator constructs converted waveforms directly by substituting source SSL representations with their nearest counterparts from a matching pool, which comprises SSL representations extracted from the target audio.
Consequently, this framework circumvents the challenge of disentanglement, effectively eliminating the issue of timbre leakage. Experimental results confirm that the proposed system delivers promising performance in the context of one-shot SVC across intra-language, cross-language, and cross-domain evaluations.

![Overall Architecture](Architecture_1.png)

## Compared Systems

- **SpkEmb-FastSVC**: FastSVC with speaker embedding extracted from ECAPA-TDNN, and linguistic features extracted from WenetSpeech.
- **NeuCo-HiFi-GAN**: neural concatenative method using the Hifi-GAN framework with a neural source-filter module as the audio synthesizer.
- **NeuCo-FastSVC**: neural concatenative method sharing the same audio synthesizer as SpkEmb-FastSVC.
- **NeuCoSVC**: our proposed neural concatenative method utilizing the FastSVC architecture and the LTV module.

## Audio Samples

<hr>

### Intra-Language

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">Source</th>
    <th class="tg-0pky">Reference</th>
    <th class="tg-0pky">SpkEmb FastSVC</th>
    <th class="tg-0pky">NeuCo-HiFi-GAN</th>
    <th class="tg-0pky">NeuCo-FastSVC</th>
    <th class="tg-0pky">NeuCoSVC(Proposed)</th>
  </tr>
</thead>
<tbody>
	<tr>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\source_audio\M8_春风十里.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\reference_audio\OpenSinger\M26.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\spkemb_fastsvc\To_OpenSinger\M8_春风十里_M26.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm-HifiGAN\To_OpenSinger\M8_春风十里_M26.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc\To_OpenSinger\M8_春风十里_M26.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc_nhv\To_OpenSinger\M8_春风十里_M26.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
	</tr>
	<tr>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\source_audio\M16_梵高先生.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\reference_audio\OpenSinger\W46.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\spkemb_fastsvc\To_OpenSinger\M16_梵高先生_W46.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm-HifiGAN\To_OpenSinger\M16_梵高先生_W46.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc\To_OpenSinger\M16_梵高先生_W46.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc_nhv\To_OpenSinger\M16_梵高先生_W46.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
	</tr>
	<tr>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\source_audio\W4_天黑黑.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\reference_audio\OpenSinger\M27.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\spkemb_fastsvc\To_OpenSinger\W4_天黑黑_M27.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm-HifiGAN\To_OpenSinger\W4_天黑黑_M27.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc\To_OpenSinger\W4_天黑黑_M27.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc_nhv\To_OpenSinger\W4_天黑黑_M27.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
	</tr>
	<tr>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\source_audio\W23_Dance.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\reference_audio\OpenSinger\W47.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\spkemb_fastsvc\To_OpenSinger\W23_Dance_W47.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm-HifiGAN\To_OpenSinger\W23_Dance_W47.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc\To_OpenSinger\W23_Dance_W47.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc_nhv\To_OpenSinger\W23_Dance_W47.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
	</tr>
</tbody>
</table>

### Cross-Language

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">Source</th>
    <th class="tg-0pky">Reference</th>
    <th class="tg-0pky">SpkEmb FastSVC</th>
    <th class="tg-0pky">NeuCo-HiFi-GAN</th>
    <th class="tg-0pky">NeuCo-FastSVC</th>
    <th class="tg-0pky">NeuCoSVC(Proposed)</th>
  </tr>
</thead>
<tbody>
	<tr>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\source_audio\M4_遇见.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\reference_audio\NUS48E\JLEE.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\spkemb_fastsvc\To_NUS48E\M4_遇见_JLEE.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm-HifiGAN\To_NUS48E\M4_遇见_JLEE.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc\To_NUS48E\M4_遇见_JLEE.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc_nhv\To_NUS48E\M4_遇见_JLEE.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
	</tr>
	<tr>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\source_audio\M18_空白格.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\reference_audio\NUS48E\MPUR.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\spkemb_fastsvc\To_NUS48E\M18_空白格_MPUR.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm-HifiGAN\To_NUS48E\M18_空白格_MPUR.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc\To_NUS48E\M18_空白格_MPUR.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc_nhv\To_NUS48E\M18_空白格_MPUR.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
	</tr>
	<tr>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\source_audio\W6_香格里拉.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\reference_audio\NUS48E\MCUR.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\spkemb_fastsvc\To_NUS48E\W6_香格里拉_MCUR.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm-HifiGAN\To_NUS48E\W6_香格里拉_MCUR.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc\To_NUS48E\W6_香格里拉_MCUR.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc_nhv\To_NUS48E\W6_香格里拉_MCUR.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
	</tr>
	<tr>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\source_audio\W29_眼泪成诗.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\reference_audio\NUS48E\SAMF.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\spkemb_fastsvc\To_NUS48E\W29_眼泪成诗_SAMF.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm-HifiGAN\To_NUS48E\W29_眼泪成诗_SAMF.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc\To_NUS48E\W29_眼泪成诗_SAMF.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc_nhv\To_NUS48E\W29_眼泪成诗_SAMF.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
	</tr>
</tbody>
</table>

### Cross-Domain

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">Source</th>
    <th class="tg-0pky">Reference</th>
    <th class="tg-0pky">SpkEmb FastSVC</th>
    <th class="tg-0pky">NeuCo-HiFi-GAN</th>
    <th class="tg-0pky">NeuCo-FastSVC</th>
    <th class="tg-0pky">NeuCoSVC(Proposed)</th>
  </tr>
</thead>
<tbody>
	<tr>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\source_audio\M14_谁.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\reference_audio\Speech\emma.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\spkemb_fastsvc\To_Speech\M14_谁_emma.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm-HifiGAN\To_Speech\M14_谁_emma.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc\To_Speech\M14_谁_emma.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc_nhv\To_Speech\M14_谁_emma.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
	</tr>
	<tr>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\source_audio\M22_当你老了.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\reference_audio\Speech\siyuanli.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\spkemb_fastsvc\To_Speech\M22_当你老了_siyuanli.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm-HifiGAN\To_Speech\M22_当你老了_siyuanli.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc\To_Speech\M22_当你老了_siyuanli.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc_nhv\To_Speech\M22_当你老了_siyuanli.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
	</tr>
	<tr>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\source_audio\W22_舞娘.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\reference_audio\Speech\haowei.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\spkemb_fastsvc\To_Speech\W22_舞娘_haowei.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm-HifiGAN\To_Speech\W22_舞娘_haowei.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc\To_Speech\W22_舞娘_haowei.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc_nhv\To_Speech\W22_舞娘_haowei.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
	</tr>
	<tr>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\source_audio\W40_易燃易爆炸.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\reference_audio\Speech\tianxia.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\spkemb_fastsvc\To_Speech\W40_易燃易爆炸_tianxia.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm-HifiGAN\To_Speech\W40_易燃易爆炸_tianxia.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc\To_Speech\W40_易燃易爆炸_tianxia.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc_nhv\To_Speech\W40_易燃易爆炸_tianxia.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
	</tr>
</tbody>
</table>

### Duration Study

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">Source</th>
    <th class="tg-0pky">Reference</th>
    <th class="tg-0pky">5s</th>
    <th class="tg-0pky">10s</th>
    <th class="tg-0pky">30s</th>
    <th class="tg-0pky">60s</th>
    <th class="tg-0pky">90s</th>
  </tr>
</thead>
<tbody>
	<tr>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\source_audio\M8_春风十里.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\reference_audio\OpenSinger\M26.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc_nhv\To_OpenSinger_5s\M8_春风十里_M26.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc_nhv\To_OpenSinger_10s\M8_春风十里_M26.wav"
					type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc_nhv\To_OpenSinger_30s\M8_春风十里_M26.wav"
					type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc_nhv\To_OpenSinger_60s\M8_春风十里_M26.wav"
					type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc_nhv\To_OpenSinger_90s\M8_春风十里_M26.wav"
					type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
	</tr>
	<tr>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\source_audio\M16_梵高先生.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\reference_audio\OpenSinger\W46.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc_nhv\To_OpenSinger_5s\M16_梵高先生_W46.wav"
					type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc_nhv\To_OpenSinger_10s\M16_梵高先生_W46.wav"
					type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc_nhv\To_OpenSinger_30s\M16_梵高先生_W46.wav"
					type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc_nhv\To_OpenSinger_60s\M16_梵高先生_W46.wav"
					type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc_nhv\To_OpenSinger_90s\M16_梵高先生_W46.wav"
					type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
	</tr>
	<tr>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\source_audio\W4_天黑黑.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\reference_audio\OpenSinger\M27.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc_nhv\To_OpenSinger_5s\W4_天黑黑_M27.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc_nhv\To_OpenSinger_10s\W4_天黑黑_M27.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc_nhv\To_OpenSinger_30s\W4_天黑黑_M27.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc_nhv\To_OpenSinger_60s\W4_天黑黑_M27.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc_nhv\To_OpenSinger_90s\W4_天黑黑_M27.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
	</tr>
	<tr>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\source_audio\W23_Dance.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\reference_audio\OpenSinger\W47.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc_nhv\To_OpenSinger_5s\W23_Dance_W47.wav"
					type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc_nhv\To_OpenSinger_10s\W23_Dance_W47.wav"
					type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc_nhv\To_OpenSinger_30s\W23_Dance_W47.wav"
					type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc_nhv\To_OpenSinger_60s\W23_Dance_W47.wav"
					type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\MOS_converted\wavlm_fastsvc_nhv\To_OpenSinger_90s\W23_Dance_W47.wav"
					type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
	</tr>
</tbody>
</table>