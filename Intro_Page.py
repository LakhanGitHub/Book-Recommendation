<<<<<<< HEAD
import gradio as gr

with gr.Blocks() as Intro:
    gr.Markdown(
        """
<h1 style="text-align: center;">✨ <strong>BookSage</strong> – <em>Your Smart Book Companion</em></h1>

## 📚 Discover Books That Truly Resonate With You
<h3>Tired of generic book lists and endless scrolling? **BookSage** uses cutting-edge AI to understand your emotions and preferences—recommending books that match your mood, interests, and the way you think.</h3>

---
<h2> 💬 Just Say What You're In the Mood For</h2>
**Whether you're feeling nostalgic, curious, inspired, or just looking for a thrilling escape, BookSage listens.**

    “I want a heartwarming story about self-discovery”
    “Something dark, suspenseful, and unforgettable”

<h2> 🧠 Powered by Modern AI (LLM)</h2> 
<ul>
   <li><b>Natural Language Input</b>: You speak like a human. Our AI listens like one</li>  
   <li><b>Deep Semantic Understanding</b>: Using advanced text embeddings (all-MiniLM-L6-v2), we analyze the true meaning of your query</li>  
   <li><b>Emotion-Aware Matching</b>: With emotion recognition (j-hartmann/emotion-english-distilroberta-base), we detect the emotional tone behind your words</li>
   <li><b>Personalized Book Recommendations</b>:Instantly get book suggestions that resonate—with your mind and mood</li> 
</ul>
<h2> 💡 Why You’ll Love It</h2> 
<ul>
   <li>🎯 <b>Emotionally aligned, deeply relevant, never boring. Not just “similar books,” but emotionally relevant stories</li>
   <li>⚡ Fast & Seamless: Results in seconds, no signup required</li>
   <li>📚 Curated With Care: Powered by real AI, not random lists</li>
   <li>🌟 Great for Everyone – From casual readers to bookworms</li>
 </ul>  
   
---
<p style="text-align">
  <a href="https://github.com/LakhanGitHub/Book-Recommendation" target="_blank" style="text-decoration:none;">
    &nbsp;<strong>View the code on GitHub</strong>
  </a>
  
  <svg xmlns="http://www.w3.org/2000/svg" x="0px" y="0px" width="100" height="100" viewBox="0 0 48 48">
<path fill="#fff" d="M41,24c0,9.4-7.6,17-17,17S7,33.4,7,24S14.6,7,24,7S41,14.6,41,24z"></path><path fill="#455a64" d="M21,41v-5.5c0-0.3,0.2-0.5,0.5-0.5s0.5,0.2,0.5,0.5V41h2v-6.5c0-0.3,0.2-0.5,0.5-0.5s0.5,0.2,0.5,0.5	V41h2v-5.5c0-0.3,0.2-0.5,0.5-0.5s0.5,0.2,0.5,0.5V41h1.8c0.2-0.3,0.2-0.6,0.2-1.1V36c0-2.2-1.9-5.2-4.3-5.2h-2.5	c-2.3,0-4.3,3.1-4.3,5.2v3.9c0,0.4,0.1,0.8,0.2,1.1H21L21,41z M40.1,26.4L40.1,26.4c0,0-1.3-0.4-2.4-0.4h-0.1	c-1.1,0-2.9,0.3-2.9,0.3c-0.1,0-0.1,0-0.1-0.1s0-0.1,0.1-0.1s2-0.3,3.1-0.3s2.4,0.4,2.5,0.4s0.1,0.1,0.1,0.2	C40.2,26.3,40.2,26.4,40.1,26.4z M39.8,27.2L39.8,27.2c0,0-1.4-0.4-2.6-0.4c-0.9,0-3,0.2-3.1,0.2S34,27,34,26.9s0-0.1,0.1-0.1	s2.2-0.2,3.1-0.2c1.3,0,2.6,0.4,2.6,0.4c0.1,0,0.1,0.1,0.1,0.2C39.9,27.1,39.9,27.2,39.8,27.2z M7.8,26.4c-0.1,0-0.1,0-0.1-0.1	s0-0.1,0.1-0.2c0.8-0.2,2.4-0.5,3.3-0.5c0.8,0,3.5,0.2,3.6,0.2s0.1,0.1,0.1,0.1c0,0.1-0.1,0.1-0.1,0.1s-2.7-0.2-3.5-0.2	C10.1,26,8.6,26.2,7.8,26.4L7.8,26.4z M8.2,27.9c0,0-0.1,0-0.1-0.1s0-0.1,0-0.2c0.1,0,1.4-0.8,2.9-1c1.3-0.2,4,0.1,4.2,0.1	c0.1,0,0.1,0.1,0.1,0.1c0,0.1-0.1,0.1-0.1,0.1l0,0c0,0-2.8-0.3-4.1-0.1C9.6,27.1,8.2,27.9,8.2,27.9L8.2,27.9z"></path><path fill="#455a64" d="M14.2,23.5c0-4.4,4.6-8.5,10.3-8.5s10.3,4,10.3,8.5S31.5,31,24.5,31S14.2,27.9,14.2,23.5z"></path><path fill="#455a64" d="M28.6,16.3c0,0,1.7-2.3,4.8-2.3c1.2,1.2,0.4,4.8,0,5.8L28.6,16.3z M20.4,16.3c0,0-1.7-2.3-4.8-2.3	c-1.2,1.2-0.4,4.8,0,5.8L20.4,16.3z M20.1,35.9c0,0-2.3,0-2.8,0c-1.2,0-2.3-0.5-2.8-1.5c-0.6-1.1-1.1-2.3-2.6-3.3	c-0.3-0.2-0.1-0.4,0.4-0.4c0.5,0.1,1.4,0.2,2.1,1.1c0.7,0.9,1.5,2,2.8,2s2.7,0,3.5-0.9L20.1,35.9z"></path><path fill="#00bcd4" d="M24,4C13,4,4,13,4,24s9,20,20,20s20-9,20-20S35,4,24,4z M24,40c-8.8,0-16-7.2-16-16S15.2,8,24,8	s16,7.2,16,16S32.8,40,24,40z"></path>
</svg>
  
</p>
"""
    )
    gr.Markdown("Made with ❤️ by Lakhan", elem_id="footer", elem_classes="footer-note")

if __name__ == "__main__":
    Intro.launch()
=======
import gradio as gr

with gr.Blocks() as Intro:
    gr.Markdown(
        """
<h1 style="text-align: center;">✨ <strong>BookSage</strong> – <em>Your Smart Book Companion</em></h1>

## 📚 Discover Books That Truly Resonate With You
<h3>Tired of generic book lists and endless scrolling? **BookSage** uses cutting-edge AI to understand your emotions and preferences—recommending books that match your mood, interests, and the way you think.</h3>

---
<h2> 💬 Just Say What You're In the Mood For</h2>
**Whether you're feeling nostalgic, curious, inspired, or just looking for a thrilling escape, BookSage listens.**

    “I want a heartwarming story about self-discovery”
    “Something dark, suspenseful, and unforgettable”

<h2> 🧠 Powered by Modern AI (LLM)</h2> 
<ul>
   <li><b>Natural Language Input</b>: You speak like a human. Our AI listens like one</li>  
   <li><b>Deep Semantic Understanding</b>: Using advanced text embeddings (all-MiniLM-L6-v2), we analyze the true meaning of your query</li>  
   <li><b>Emotion-Aware Matching</b>: With emotion recognition (j-hartmann/emotion-english-distilroberta-base), we detect the emotional tone behind your words</li>
   <li><b>Personalized Book Recommendations</b>:Instantly get book suggestions that resonate—with your mind and mood</li> 
</ul>
<h2> 💡 Why You’ll Love It</h2> 
<ul>
   <li>🎯 <b>Emotionally aligned, deeply relevant, never boring. Not just “similar books,” but emotionally relevant stories</li>
   <li>⚡ Fast & Seamless: Results in seconds, no signup required</li>
   <li>📚 Curated With Care: Powered by real AI, not random lists</li>
   <li>🌟 Great for Everyone – From casual readers to bookworms</li>
 </ul>  
   
---
<p style="text-align">
  <a href="https://github.com/LakhanGitHub/Book-Recommendation" target="_blank" style="text-decoration:none;">
    &nbsp;<strong>View the code on GitHub</strong>
  </a>
  
  <svg xmlns="http://www.w3.org/2000/svg" x="0px" y="0px" width="100" height="100" viewBox="0 0 48 48">
<path fill="#fff" d="M41,24c0,9.4-7.6,17-17,17S7,33.4,7,24S14.6,7,24,7S41,14.6,41,24z"></path><path fill="#455a64" d="M21,41v-5.5c0-0.3,0.2-0.5,0.5-0.5s0.5,0.2,0.5,0.5V41h2v-6.5c0-0.3,0.2-0.5,0.5-0.5s0.5,0.2,0.5,0.5	V41h2v-5.5c0-0.3,0.2-0.5,0.5-0.5s0.5,0.2,0.5,0.5V41h1.8c0.2-0.3,0.2-0.6,0.2-1.1V36c0-2.2-1.9-5.2-4.3-5.2h-2.5	c-2.3,0-4.3,3.1-4.3,5.2v3.9c0,0.4,0.1,0.8,0.2,1.1H21L21,41z M40.1,26.4L40.1,26.4c0,0-1.3-0.4-2.4-0.4h-0.1	c-1.1,0-2.9,0.3-2.9,0.3c-0.1,0-0.1,0-0.1-0.1s0-0.1,0.1-0.1s2-0.3,3.1-0.3s2.4,0.4,2.5,0.4s0.1,0.1,0.1,0.2	C40.2,26.3,40.2,26.4,40.1,26.4z M39.8,27.2L39.8,27.2c0,0-1.4-0.4-2.6-0.4c-0.9,0-3,0.2-3.1,0.2S34,27,34,26.9s0-0.1,0.1-0.1	s2.2-0.2,3.1-0.2c1.3,0,2.6,0.4,2.6,0.4c0.1,0,0.1,0.1,0.1,0.2C39.9,27.1,39.9,27.2,39.8,27.2z M7.8,26.4c-0.1,0-0.1,0-0.1-0.1	s0-0.1,0.1-0.2c0.8-0.2,2.4-0.5,3.3-0.5c0.8,0,3.5,0.2,3.6,0.2s0.1,0.1,0.1,0.1c0,0.1-0.1,0.1-0.1,0.1s-2.7-0.2-3.5-0.2	C10.1,26,8.6,26.2,7.8,26.4L7.8,26.4z M8.2,27.9c0,0-0.1,0-0.1-0.1s0-0.1,0-0.2c0.1,0,1.4-0.8,2.9-1c1.3-0.2,4,0.1,4.2,0.1	c0.1,0,0.1,0.1,0.1,0.1c0,0.1-0.1,0.1-0.1,0.1l0,0c0,0-2.8-0.3-4.1-0.1C9.6,27.1,8.2,27.9,8.2,27.9L8.2,27.9z"></path><path fill="#455a64" d="M14.2,23.5c0-4.4,4.6-8.5,10.3-8.5s10.3,4,10.3,8.5S31.5,31,24.5,31S14.2,27.9,14.2,23.5z"></path><path fill="#455a64" d="M28.6,16.3c0,0,1.7-2.3,4.8-2.3c1.2,1.2,0.4,4.8,0,5.8L28.6,16.3z M20.4,16.3c0,0-1.7-2.3-4.8-2.3	c-1.2,1.2-0.4,4.8,0,5.8L20.4,16.3z M20.1,35.9c0,0-2.3,0-2.8,0c-1.2,0-2.3-0.5-2.8-1.5c-0.6-1.1-1.1-2.3-2.6-3.3	c-0.3-0.2-0.1-0.4,0.4-0.4c0.5,0.1,1.4,0.2,2.1,1.1c0.7,0.9,1.5,2,2.8,2s2.7,0,3.5-0.9L20.1,35.9z"></path><path fill="#00bcd4" d="M24,4C13,4,4,13,4,24s9,20,20,20s20-9,20-20S35,4,24,4z M24,40c-8.8,0-16-7.2-16-16S15.2,8,24,8	s16,7.2,16,16S32.8,40,24,40z"></path>
</svg>
  
</p>
"""
    )
    gr.Markdown("Made with ❤️ by Lakhan", elem_id="footer", elem_classes="footer-note")

if __name__ == "__main__":
    Intro.launch()
>>>>>>> 88cb0c7b905e8588877d3a61ab45044cc4a2961e
