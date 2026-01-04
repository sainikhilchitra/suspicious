<h2>Abstract</h2>
<p>
Video anomaly detection in surveillance environments is challenging due to the scarcity of abnormal events,
lack of frame-level annotations, and the subtle nature of behaviors such as theft.
Prediction-based approaches address this problem by learning normal spatio-temporal patterns through
future-frame prediction and detecting anomalies as deviations between predicted and actual frames.
However, most existing methods rely heavily on pixel-level prediction errors, which are sensitive to noise
and often fail to capture localized and subtle abnormal activities.
</p>

<p>
To address these limitations, we propose an <b>attention-enhanced future-frame prediction framework</b>
for unsupervised video anomaly detection.
The proposed model integrates a spatio-temporal attention mechanism into a CNN–ConvLSTM–decoder
architecture, enabling the network to focus on behavior-relevant regions and critical temporal segments.
In addition, anomaly scoring is performed using a combination of pixel-level and feature-level discrepancies,
resulting in improved robustness to noise and enhanced sensitivity to subtle theft-like behaviors.
The model is trained solely on normal data and does not require explicit anomaly annotations.
</p>

<hr/>

<h2>Additional / Unique Features Added</h2>

<h3>1. Spatio-Temporal Attention Mechanism</h3>
<p>
Most prediction-based anomaly detection models treat all spatial regions and temporal frames equally.
In practice, abnormal behaviors such as theft are localized (e.g., hand–object interactions) and occur
over short time intervals.
</p>

<p>
The proposed spatio-temporal attention module assigns higher importance to:
</p>
<ul>
  <li>Regions with significant motion or interaction</li>
  <li>Temporal segments where behavior changes abruptly</li>
</ul>

<p>
By suppressing static background regions and emphasizing behavior-relevant areas,
the attention mechanism enables the model to concentrate its predictive capacity where anomalies are
most likely to occur.
</p>

<h3>2. Feature-Level Anomaly Scoring</h3>
<p>
Existing methods primarily rely on pixel-level metrics (e.g., PSNR or MSE) to compute anomaly scores.
Such metrics are sensitive to illumination changes, camera noise, and minor background variations.
</p>

<p>
To overcome this limitation, the proposed framework introduces feature-level discrepancy by comparing
deep representations of predicted and actual frames.
These representations encode semantic and motion information, making them more robust to noise.
</p>

<p>
The final anomaly score is computed by combining:
</p>
<ul>
  <li>Pixel-level prediction error (captures large deviations)</li>
  <li>Feature-level discrepancy (captures subtle behavioral changes)</li>
</ul>

<hr/>

<h2>Effect of the Added Features</h2>

<ul>
  <li><b>Improved Sensitivity to Subtle Anomalies:</b>
      Attention highlights small but meaningful motion patterns that pixel-level errors may overlook.</li>

  <li><b>Reduced False Positives:</b>
      Background motion and illumination changes are downweighted, leading to fewer spurious detections.</li>

  <li><b>Better Behavioral Modeling:</b>
      Feature-level comparison captures semantic inconsistencies rather than raw pixel noise.</li>

  <li><b>Lightweight Design:</b>
      The added modules introduce minimal computational overhead compared to memory banks or diffusion models.</li>
</ul>

<hr/>

<h2>Architecture Explanation</h2>

<h3>Overall Pipeline</h3>
<pre>
Input Past Frames
      ↓
CNN Encoder
      ↓
ConvLSTM
      ↓
Spatio-Temporal Attention
      ↓
Decoder (Future Frame Prediction)
      ↓
Pixel-Level + Feature-Level Comparison
      ↓
Anomaly Score
</pre>

<h3>Component-wise Description</h3>

<h4>CNN Encoder</h4>
<p>
The CNN encoder processes each input frame independently and extracts spatial features such as edges,
object shapes, and local motion cues.
This step transforms raw pixel input into a compact and semantically meaningful representation.
</p>

<h4>ConvLSTM</h4>
<p>
ConvLSTM layers model the temporal evolution of spatial features across consecutive frames.
They learn normal motion dynamics and interaction patterns present in surveillance scenes.
Abnormal events disrupt these learned temporal regularities.
</p>

<h4>Spatio-Temporal Attention Module</h4>
<p>
The attention module operates on ConvLSTM feature maps and assigns adaptive weights across spatial
and temporal dimensions.
Regions and frames contributing more to behavioral changes receive higher weights,
guiding the decoder toward important motion patterns.
</p>

<h4>Decoder</h4>
<p>
The decoder reconstructs the predicted future frame using attended spatio-temporal features.
Accurate prediction indicates normal behavior, while poor prediction indicates abnormal activity.
</p>

<h4>Anomaly Scoring</h4>
<p>
During inference, the predicted frame is compared with the actual frame at both pixel and feature levels.
The combined discrepancy serves as the anomaly score, with higher values indicating abnormal events.
</p>

<hr/>

<h2>Literature Survey</h2>

<table border="1" cellpadding="6" cellspacing="0">
<tr>
  <th>Title</th>
  <th>Author & Year</th>
  <th>Approach</th>
  <th>Dataset</th>
  <th>Results</th>
  <th>Limitations</th>
</tr>

<tr>
  <td>Future Frame Prediction for Anomaly Detection</td>
  <td>Liu et al., 2018</td>
  <td>Prediction + GAN</td>
  <td>UCSD, Avenue</td>
  <td>AUC 95.4% (Ped2)</td>
  <td>Pixel-level PSNR, blurry predictions</td>
</tr>

<tr>
  <td>Future Frame Prediction Network for VAD</td>
  <td>Zhu et al., 2019</td>
  <td>CNN + ConvLSTM</td>
  <td>UCSD, Avenue</td>
  <td>AUC 95.2% (Ped2)</td>
  <td>Weak long-term modeling</td>
</tr>

<tr>
  <td>Memory-Augmented Frame Prediction</td>
  <td>Liu et al., 2020</td>
  <td>Prediction + Memory</td>
  <td>UCSD</td>
  <td>AUC 96.1% (Ped2)</td>
  <td>High computational cost</td>
</tr>

<tr>
  <td>Spatio-Temporal Prediction and Reconstruction</td>
  <td>Ting Liu et al., 2022</td>
  <td>Prediction + Reconstruction</td>
  <td>UCSD, Avenue</td>
  <td>AUC 96.6% (Ped2)</td>
  <td>Heavy architecture</td>
</tr>

<tr>
  <td>Object Relationship-Based VAD</td>
  <td>Wang et al., 2023</td>
  <td>Attention-based Reconstruction</td>
  <td>UCSD, Avenue</td>
  <td>AUC 98.4% (Ped2)</td>
  <td>Reconstruction bias</td>
</tr>
</table>
