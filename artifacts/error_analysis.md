# Error analysis (UK test)

## Setup risks
- AG `World` ↔ BBC `politics` is an imperfect label bridge.
- BBC items are often article bodies; we use title if present, else first sentence / first ~16 tokens as a headline proxy.
- UK data is small; Stage-2 uses early stopping on UK val only (test stays frozen).

## What the before model tends to miss
Stage-1 is trained on US-centric AG News titles. On UK BBC headlines it often misses:
- British politics entities (Westminster, party names, ministers)
- UK sport (Premier League clubs, cricket, rugby)
- UK spellings / local brands in tech/business copy

Of 20 sampled Stage-1 errors, 15 were corrected after UK fine-tuning and 5 remained wrong in this sample.

### Corrected after UK fine-tune (examples)
- **true=politics** before=`business` → after=`politics`: new drink limit would cut toll more lives than previously thought could be saved by cutting
- **true=politics** before=`tech` → after=`politics`: msps hear renewed climate warning climate change could be completely out of control within several decades
- **true=business** before=`tech` → after=`business`: fbi agent colludes with analyst a former fbi agent and an internet stock picker have been
- **true=politics** before=`business` → after=`politics`: minimum rate for foster parents foster carers are to be guaranteed a minimum allowance to help
- **true=politics** before=`tech` → after=`politics`: teens know little of politics teenagers questioned for a survey have shown little interest in politics
- **true=business** before=`politics` → after=`business`: libya takes $1bn in unfrozen funds libya has withdrawn $1bn in assets from the us assets
- **true=politics** before=`tech` → after=`politics`: howard backs stem cell research michael howard has backed stem cell research saying it is important
- **true=politics** before=`business` → after=`politics`: army chiefs in regiments decision military chiefs are expected to meet to make a final decision

### Still wrong after fine-tune (examples)
- **true=politics** before=`business` after=`business`: jowell confirms casino climbdown tessa jowell has announced plans to limit the number of new casinos
- **true=business** before=`tech` after=`tech`: bt offers equal access to rivals bt has moved to pre-empt a possible break-up of its
- **true=politics** before=`tech` after=`business`: uk firms embracing e-commerce uk firms are embracing internet trading opportunities as never before e-commerce minister
- **true=politics** before=`business` after=`business`: uk will stand firm on eu rebate britain s £3bn eu rebate is not up for
- **true=politics** before=`business` after=`business`: baa support ahead of court battle uk airport operator baa has reiterated its support for the
