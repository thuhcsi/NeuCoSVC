<!-- # Neural Concatenative Singing Voice Conversion -->

## Abstract

Any-to-any singing voice conversion (SVC) is confronted with a significant challenge of "timbre leakage" issue caused by insufficient disentanglement between the content and the speaker timbre. To address the formidable challenge of disentanglement, this study introduces a novel neural concatenative singing voice conversion (NeuCoSVC) framework. The NeuCoSVC framework comprises a self-supervised learning (SSL) representation extractor, a neural harmonic signal generator, and a waveform synthesizer. 
The SSL model condenses the audio into a sequence of fixed-dimensional SSL features. The harmonic signal generator produces both raw and filtered harmonic signals leveraging a linear time-varying filter given condition features. Simultaneously, the audio generator creates waveforms directly from the SSL features, integrating both the harmonic signals and the loudness. During inference, the audio generator constructs converted waveforms directly by substituting source SSL representations with their nearest counterparts from a matching pool, which comprises SSL representations extracted from the target audio.
Consequently, this framework circumvents the challenge of disentanglement, effectively eliminating the issue of timbre leakage. Experimental results confirm that the proposed system delivers promising performance in the context of one-shot SVC across intra-language, cross-language, and cross-domain evaluations.

<div style="text-align: center;">
	<img src="/NeuCoSVC/Architecture_1.png" alt="Overall Architecture" style="max-width: 60%">
</div>

## Compared Systems

- **SpkEmb-FastSVC**: speaker embedding with FastSVC as the audio synthesizer. The speaker embeddings are extracted from ECAPA-TDNN[<sup>1</sup>](#references), and the linguistic features are extracted from WenetSpeech[<sup>2</sup>](#references).
- **NeuCoSVC (Proposed)**: the proposed system, consisting of neural concatenative method with the FastSVC architecture and the LTV harmonic filter module.
- **NeuCo-FastSVC**: neural concatenative method with FastSVC as the audio synthesizer.
- **NeuCo-HiFi-GAN**: neural concatenative method with nsf-HiFi-GAN[<sup>3,4</sup>](#references) as the audio synthesizer.

## Audio Samples

We conduct any-to-any SVC experiments in three different scenarios: **intra-language** and **cross-language** conversions for in-domain tests, and intra-language conversions for **cross-domain** evaluation. In the in-domain SVC, target speakers' singing voices are employed, whereas, in the cross-domain SVC, their speech voices serve as input. All speakers from the reference utterance remain unseen during training.

Note that the reference audio in intra-/cross-language scenarios is approximately 10 minutes long, while the one in the cross-domain scenario is about 30 seconds long, as mentioned in Section 4 of our paper. In the demo web, we've only included a single segment of around 10 seconds of the target person's audio to demonstrate their voice characteristics. Additionally, singing audio of varying lengths are used in the **duration study** to assess the impact of reference audio duration on conversion quality.

Please feel free to explore the demo and refer to our paper for more detailed information on the experimental setup and results.

<hr>

### Intra-Language

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">Source</th>
    <th class="tg-0pky">Reference</th>
    <th class="tg-0pky">SpkEmb-FastSVC</th>
    <th class="tg-0pky">NeuCoSVC (Proposed)</th>
    <th class="tg-0pky">NeuCo-FastSVC</th>
    <th class="tg-0pky">NeuCo-HiFi-GAN</th>
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
				<source src="audios\reference_audio\OpenSinger\M26.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\spkemb_fastsvc\To_OpenSinger\M4_遇见_M26.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc_nhv\To_OpenSinger\M4_遇见_M26.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc\To_OpenSinger\M4_遇见_M26.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_HiFiGAN\To_OpenSinger\M4_遇见_M26.wav" type="audio/mpeg">
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
				<source src="audios\converted\spkemb_fastsvc\To_OpenSinger\M16_梵高先生_W46.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc_nhv\To_OpenSinger\M16_梵高先生_W46.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc\To_OpenSinger\M16_梵高先生_W46.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_HiFiGAN\To_OpenSinger\M16_梵高先生_W46.wav" type="audio/mpeg">
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
				<source src="audios\converted\spkemb_fastsvc\To_OpenSinger\W4_天黑黑_M27.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc_nhv\To_OpenSinger\W4_天黑黑_M27.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc\To_OpenSinger\W4_天黑黑_M27.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_HiFiGAN\To_OpenSinger\W4_天黑黑_M27.wav" type="audio/mpeg">
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
				<source src="audios\reference_audio\OpenSinger\W47.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\spkemb_fastsvc\To_OpenSinger\W29_眼泪成诗_W47.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc_nhv\To_OpenSinger\W29_眼泪成诗_W47.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc\To_OpenSinger\W29_眼泪成诗_W47.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_HiFiGAN\To_OpenSinger\W29_眼泪成诗_W47.wav" type="audio/mpeg">
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
    <th class="tg-0pky">SpkEmb-FastSVC</th>
    <th class="tg-0pky">NeuCoSVC (Proposed)</th>
    <th class="tg-0pky">NeuCo-FastSVC</th>
    <th class="tg-0pky">NeuCo-HiFi-GAN</th>
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
				<source src="audios\converted\spkemb_fastsvc\To_NUS48E\M4_遇见_JLEE.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc_nhv\To_NUS48E\M4_遇见_JLEE.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc\To_NUS48E\M4_遇见_JLEE.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_HiFiGAN\To_NUS48E\M4_遇见_JLEE.wav" type="audio/mpeg">
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
				<source src="audios\reference_audio\NUS48E\MPUR.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\spkemb_fastsvc\To_NUS48E\M16_梵高先生_MPUR.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc_nhv\To_NUS48E\M16_梵高先生_MPUR.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc\To_NUS48E\M16_梵高先生_MPUR.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_HiFiGAN\To_NUS48E\M16_梵高先生_MPUR.wav" type="audio/mpeg">
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
				<source src="audios\reference_audio\NUS48E\MCUR.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\spkemb_fastsvc\To_NUS48E\W4_天黑黑_MCUR.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc_nhv\To_NUS48E\W4_天黑黑_MCUR.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc\To_NUS48E\W4_天黑黑_MCUR.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_HiFiGAN\To_NUS48E\W4_天黑黑_MCUR.wav" type="audio/mpeg">
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
				<source src="audios\converted\spkemb_fastsvc\To_NUS48E\W29_眼泪成诗_SAMF.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc_nhv\To_NUS48E\W29_眼泪成诗_SAMF.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc\To_NUS48E\W29_眼泪成诗_SAMF.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_HiFiGAN\To_NUS48E\W29_眼泪成诗_SAMF.wav" type="audio/mpeg">
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
    <th class="tg-0pky">SpkEmb-FastSVC</th>
    <th class="tg-0pky">NeuCoSVC (Proposed)</th>
    <th class="tg-0pky">NeuCo-FastSVC</th>
    <th class="tg-0pky">NeuCo-HiFi-GAN</th>
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
				<source src="audios\reference_audio\Speech\emma.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\spkemb_fastsvc\To_Speech\M4_遇见_emma.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc_nhv\To_Speech\M4_遇见_emma.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc\To_Speech\M4_遇见_emma.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_HiFiGAN\To_Speech\M4_遇见_emma.wav" type="audio/mpeg">
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
				<source src="audios\reference_audio\Speech\honglunzhang.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\spkemb_fastsvc\To_Speech\M16_梵高先生_honglunzhang.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc_nhv\To_Speech\M16_梵高先生_honglunzhang.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc\To_Speech\M16_梵高先生_honglunzhang.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_HiFiGAN\To_Speech\M16_梵高先生_honglunzhang.wav" type="audio/mpeg">
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
				<source src="audios\reference_audio\Speech\siyuanli.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\spkemb_fastsvc\To_Speech\W4_天黑黑_siyuanli.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc_nhv\To_Speech\W4_天黑黑_siyuanli.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc\To_Speech\W4_天黑黑_siyuanli.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_HiFiGAN\To_Speech\W4_天黑黑_siyuanli.wav" type="audio/mpeg">
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
				<source src="audios\reference_audio\Speech\tianxia.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\spkemb_fastsvc\To_Speech\W29_眼泪成诗_tianxia.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc_nhv\To_Speech\W29_眼泪成诗_tianxia.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc\To_Speech\W29_眼泪成诗_tianxia.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_HiFiGAN\To_Speech\W29_眼泪成诗_tianxia.wav" type="audio/mpeg">
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
				<source src="audios\source_audio\M4_遇见.wav" type="audio/mpeg">
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
				<source src="audios\converted\wavlm_fastsvc_nhv\5s\M4_遇见_M26.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc_nhv\10s\M4_遇见_M26.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc_nhv\30s\M4_遇见_M26.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc_nhv\60s\M4_遇见_M26.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc_nhv\90s\M4_遇见_M26.wav" type="audio/mpeg">
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
				<source src="audios\converted\wavlm_fastsvc_nhv\5s\M16_梵高先生_W46.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc_nhv\10s\M16_梵高先生_W46.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc_nhv\30s\M16_梵高先生_W46.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc_nhv\60s\M16_梵高先生_W46.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc_nhv\90s\M16_梵高先生_W46.wav" type="audio/mpeg">
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
				<source src="audios\converted\wavlm_fastsvc_nhv\5s\W4_天黑黑_M27.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc_nhv\10s\W4_天黑黑_M27.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc_nhv\30s\W4_天黑黑_M27.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc_nhv\60s\W4_天黑黑_M27.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc_nhv\90s\W4_天黑黑_M27.wav" type="audio/mpeg">
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
				<source src="audios\reference_audio\OpenSinger\W47.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc_nhv\5s\W29_眼泪成诗_W47.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc_nhv\10s\W29_眼泪成诗_W47.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc_nhv\30s\W29_眼泪成诗_W47.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc_nhv\60s\W29_眼泪成诗_W47.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
		<td class="tg-0pky">
			<audio controls>
				<source src="audios\converted\wavlm_fastsvc_nhv\90s\W29_眼泪成诗_W47.wav" type="audio/mpeg">
				Your browser does not support this audio format.
			</audio>
		</td>
	</tr>
</tbody>
</table>

## References

[1] B. Desplanques, J. Thienpondt, and K. Demuynck, “ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification,” in Proc. Interspeech, 2020, pp. 3830–3834.

[2] B. Zhang, H. Lv, P. Guo, Q. Shao, C. Yang, L. Xie, X. Xu, H. Bu, X. Chen, C. Zeng et al., “Wenetspeech: A 10000+ hours multi-domain mandarin corpus for speech recognition,” in ICASSP. IEEE, 2022, pp. 6182–6186.

[3] X. Wang, S. Takaki, and J. Yamagishi, “Neural source-filter waveform models for statistical parametric speech synthesis,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 28, pp. 402–415, 2020.

[4] J. Kong, J. Kim, and J. Bae, “Hifi-gan: Generative adversarial networks for efficient and high fidelity speech synthesis,” in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 17 022–17 033.
