<p align="center">

  <h2 align="center">MagicPose4D: Crafting Articulated Models <br>with Appearance and Motion Control</h2>
  <p align="center">
    <a href="https://haoz19.github.io/"><strong>Hao Zhang</strong></a><sup>1*</sup>
    ·  
    <a href="https://boese0601.github.io/"><strong>Di Chang</strong></a><sup>2*</sup>
    ·
    <a href="https://www.linkedin.com/in/fang-li-8ab696223/"><strong>Fang Li</strong></a><sup>1</sup>
    ·
    <a href="https://www.ihp-lab.org/"><strong>Mohammad Soleymani</strong></a><sup>2<span>&#8224;</span></sup>
    ·
    <a href="https://vision.ai.illinois.edu/narendra-ahuja/"><strong>Narendra Ahuja</strong></a><sup>1<span>&#8224;</span></sup>
    <br>
    <sup>1</sup>University of Illinois Urbana-Champaign &nbsp;&nbsp;&nbsp; <sup>2</sup>University of Southern California
    <br>
    <sup>*</sup>Equal Contribution &nbsp;&nbsp;&nbsp; <sup><span>&#8224;</span></sup>Equal Advising
    <br>
    </br>
        <a href="">
        <img src='https://img.shields.io/badge/arXiv-MagicPose4D-green' alt='Paper PDF'>
        </a>
        <a href='https://boese0601.github.io/magicpose4d/'>
        <img src='https://img.shields.io/badge/Project_Page-MagicPose4D-blue' alt='Project Page'></a>
        <!-- <a href='https://youtu.be/VPJe6TyrT-Y'>
        <img src='https://img.shields.io/badge/YouTube-MagicPose-rgb(255, 0, 0)' alt='Youtube'></a> -->
     </br>
    <table align="center">
        <img src="./figures/hiphop-1-humanoid.gif">
    </table>
</p>

*We introduce MagicPose4D, a novel framework for 4D generation providing more accurate and customizable 4D motion retargeting. We propose a dual-phase reconstruction process that initially uses accurate 2D and pseudo 3D supervision without skeleton constraints, and subsequently refines the model with skeleton constraints to ensure physical plausibility. We incorporate a novel Global-Local Chamfer loss function that aligns the overall distribution of mesh vertices with the supervision and maintains part-level alignment without additional annotations. Our method enables cross-category motion transfer using a kinematic-chain-based skeleton, ensuring smooth transitions between frames through dynamic rigidity and achieving robust generalization without the need for additional training.*

*For 2D video motion retargeting and animation, please also check our previous work <a href="https://github.com/Boese0601/MagicDance">MagicPose</a>!*


## News
* **[2023.11.18]** Release MagicPose4D paper and project page.
