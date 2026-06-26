LLMs, GPUs, CPUs, RAM, VRAM, DISK



Latency is a major factor in LLM performance as you might know, yes you have hardware level latency but you have also software level latency.



Here is where im going with this, you might have the best hardware possible in the world right, but if your software is dog shit, your hardware is completly inefficient by this i mean:



1. im doing way more memory operation i should be doing

2. not fully utilising compute

3. having a lot of sync, halt, kernel waits, inefficient memory design, inefficient compute

4. inefficient cache design, compute not well distributed, would be faster on X

5. latency badly optimized

6. not optimized for old hardware, utilise "new features because its top tech coolness", leaving "old normal pleb hardware behind"



You have access to the entire internet and far more knowledge than me since you are a realtime database compression algorithm in some sense.



Here is what our task going to be:



We are going to analyse, find, redesign and rethink part of LLMs to create a boosted version, and this, without scaling down or losing precision or using better hardware.



First, we are going to boost efficiency by design a new system that nobody can think about by thinking at the problem rather than "trying stuff out".



Why is it slow? Hardware level is not an excuse, we are on the SOFTWARE LEVEL.



The second problem we will solve will be regarding limitation with "model size". I understand that loading model in VRAM is efficient, but VRAM is not a all or nothing problem my friend.



Lastly, PCIE gen 3 is as fast as PCIE gen 5, I ran multiple benchmarks, I have never seen an LLM pull more than 100mb/s over a PCIE lane, never in my life, cuda is not "omg this is so cool" deel, amd exist, other alternative exist, i do understand that "nvlink" give good advantage for hardware latency for cross cards operations but, i dont give a fuck about hardware thats not our job, i only care about the software layer here.



PCIE gen 5 is totally useless, it work fine on PCIE gen 3 its fast enough even you could probably run it as fast on PCIE gen 1 if its all bandwidth. Bandwidth only matter when loading the model in and even that, i never seen it go up then 200mb/s so we are not fully utilising our hardware here at all. This is what I want to make you realize.



CPU is very very good at tiny latency operation, gpu is good at ultra parrallelism.

Do you see this problem here? people just say well yeah we gotta load everything on gpu because gpu is so fast!


Thats not fucking true at 100%

im not saying you dont need gpus not at all.
im saying people dont even understand the real problem of LLM.

Also, here, we are not talking about training, training is more complex and this is not what we are trying to improve here. 


small side note here, do not be stuck about my shit saying gen 5 is useless? my guy you are lost i dont give a fuck about disk performance, all i was saying is that marketing around "SO FUCKING FAST GEN 5 BANDWIDTH OMG YOU NEED THIS" is complete fluff, you could put a gen3 only card such as a 2080ti and it would load as "fast" as on pcie gen 5.


next thnig that you neglected, kernel operations.



kernel can stall your program from memory operations and non optimisation. the best example i can give you is this:



imagine an http server running in rust ok you might get 100kreq/s, bypass the kernel or give your own memory instruction and use dpdk and now you end up with 2mil req/s.



for example, unoptimized memory operation, clearing memory, writing, reading, can stall in multithreading.



i know a lot of program that use atomic badly, that make their multithreading performance lower than single thread only because it was badly designed, even if you performance analyze something, you will not see cpu halts or gpu halts because its waiting on something, it wont show up in most used code, its usually hidden deep at like 0.001% of the program cycle, it stall because the operation is extremly slow and halt.


Add whats already proven, what is novel, what is not proven, what people have tried before and didnt work, what works.

Im mostly interested in whats novel that nobody did before, we arent talking about a tiny change here, we are talking about a redesign, an hybrid redesign.

And lastly, do we really need to load 800gb in vram? I don't think so. Pretty sure it could be efficient without 800gb of vram.

Also, what about old hardware picture this, 8x 2080ti, ok yes it got slower GDDR6 but man its GDDR6 lmao, and 8x 2080ti have more raw compute than a single 5090 in theory if you check the SM cores, idk about tensors doe but tensor is not all or nothing.

Literally nothing should stop me from running a model on only 8x 2080ti even if the model is 800gb. (If i have the ram, and the disk required of course.)


Ever though of this: Picture this, I want to run a 800gb model. I have 32x 2080ti on exactly 4 system, 8 per system. Right now without nvidia fancy tech that is only avalible for people paying 1000000000$ this is impossible to use such design, because their "driver dont allow you" this is 100% false, we can code shit software ourself. we dont need to rely on someone proprietary tech to do thing. amd doesnt even have it. the software simply dont exist yet. picture this, i want to load a 800gb model. right now software say: "nope impossible". and im telling you "yes possible." how? maybe theres more efficient way but look: user prompt -> system 1 load part A of the weights, system 2 load part B, etc user prompt goes from system 1 to system 4 until the prompt is fully done, you will barely lose latency if done correcly and it might even be faster than using a single gpu because theorically the raw compute is faster on more card, what you dont try to do: hit all the card at the same time, this will kill you, if you can do full pass system 1, full pass system 2, full pass system 3, etc until complete, finalize, etc you just solved a massive problem. add this to phase 1 because you didnt think of this.