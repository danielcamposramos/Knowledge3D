Welcome to the Deep Dive. We
are thrilled to have you with
us today because we are looking
at a stack of sources that
fundamentally challenges, well,
basically everything we think
we know about the current
trajectory of technology.
It really does.
Yeah. I want you to imagine
just for a moment a pretty mind
-bending premise.
What if the way we currently
build artificial intelligence
and, frankly, the way we store
digital knowledge across the
entire Internet is
fundamentally broken?
Right.
What if our current methods are
leading to massive, unsustain
able carbon waste and trapping
our collective human knowledge
inside these opaque, unreadable
black box machines?
It is a provocative question,
and it is exactly what the
architects behind our source
material today are asking.
We are examining a proposed
solution to this massive
structural problem.
The solution is called PMKR,
which stands for Procedural
Memory Knowledge Representation
, and we'll be looking closely
at its reference.
Especially at its reference
implementation, a system known
as Knowledge 3D or K3D.
Okay. Let's unpack this because
the stack of sources we have in
front of us today is incredibly
dense.
Very dense, yeah.
It's fascinating.
Yeah.
But it is highly consequential.
We're looking at official W3C
community group proposals.
We've got technical white
papers from a group called Ecos
ystems AI Studios.
Right.
We have deeply detailed carbon
blueprint projections,
mathematical foundation papers,
and architectural manifestos
that read less like standard
technical manuals and more like
a massive call to action for
the future of the web.
And our mission for this deep
dive is to decode all of it for
you. We are going to explore
how storing knowledge as execut
able procedures, rather than the
static files we are also used
to, can solve what the authors
call the AI duplication crisis.
Exactly.
We are going to look at how
this architecture creates a
shared, perceivable reality for
both humans and machines, and
perhaps most impactively.
We will examine the
mathematical models projecting
how this shift could
potentially eliminate 12 gigat
ons of carbon emissions over the
next decade.
12 gigatons. I mean, that's
just massive. And I want to
speak directly to you, the
listener, right now. Whether
you build the screens we look
at, whether you're navigating
the changing landscape of web
standards, or you're just a
citizen worried about the
massive energy grid required to
run the AI we use every day,
this shifts the ground under
all our feet.
It really does touch everything
.
So we need to understand the
baseline. The sources refer to
our current situation as a
knowledge duplication crisis.
But before we get into the
crisis part, what exactly is
the flaw of how we store things
right now? What is a static
payload?
The core issue identified in
these papers is that modern
digital systems suffer from
profound, pervasive duplication
. When we talk about a static
payload, we meet a piece of
data that is fixed, pre-rend
ered, and unchanging.
Like a file on my desktop.
Exactly. Think of an image file
, a text document, or a font
file.
The PMKR problem statement
points out that right now, the
exact same piece of conceptual
knowledge is stored six or more
times across entirely different
isolated systems all over the
globe.
I know the white papers use a
specific analogy to illustrate
this. Something about the
letter A. Can you walk us
through that? Because when I
think of the letter A, it just
seems like a given. It's just a
letter on my screen.
It seems simple to us, but let
's break down how a computer
actually handles it. Think
about the letter A. It is a
single, fundamental, universal
concept. But how does our
digital world store it?
Right.
First, you have the visual font
file. The true type or open
type file. This is a static
payload that tells the screen
exactly how to draw the curves,
the crossbar, the serifs of the
letter. It's a localized file
just for visual rendering.
Okay. So that's the visual part
. But my computer also needs to
know what the letter actually
is, not just what it looks like
.
Precisely. So completely
separate from that visual file,
you have the Unicode data. This
is the underlying digital
identifier, essentially a
standardized number that tells
the operating system, hey, this
is the Latin character A.
Okay.
That's stored in a different
database handled by a different
subsystem. And the frag
mentation doesn't stop there.
Let me guess. Accessibility.
Exactly. If you need
accessibility metadata, like
the phonetic rules for how a
screen reader should pronounce
that letter aloud for a
visually impaired user, or the
tactile output required to push
up the pins on a braille
display.
Right.
That information is stored in
yet another isolated static sil
o.
And now we have to factor in
the AI boom.
That is where the duplication
goes from inefficient to
critical. In the age of AI, we
have semantic training embed
dings.
Okay. Yeah.
The large language models have
their own massive high
dimensional vector
representations of what the
letter A means.
Wait, hold on. We throw around
terms like high dimensional
vector representations a lot.
But for someone listening who
hasn't built a neural network,
what does that actually mean?
Why does an AI need a vector
for a letter?
It's a great question. A
computer doesn't understand
meaning the way we do. To teach
an AI what a word or a letter
means, engineers turn that
concept into a list of numbers,
a vector.
It's a long string of numbers.
Right. And high dimensional
just means that list of numbers
is incredibly long, sometimes
thousands of numbers long for a
single concept.
These numbers map the
relationship of that letter to
every other concept the AI
knows. Does A usually come
before B? Is it a vowel? Is it
an indefinite article? All of
that meaning is baked into this
giant mathematical list.
So just to recap the sheer
scale of the waste here. For
the single fundamental concept
of the first letter of the
alphabet, we have the visual
form in a font file, the
digital identifier in a Unicode
database, the phonetic rules in
an accessibility engine, and a
massive list of mathematical
numbers in an AI's semantic
brain.
Yes. And all of these are
completely disconnected, dupl
icated, and managed separately
by different companies on
different servers.
Exactly. And Daniel Campos R
amos, the primary architect of
the system, points out in his
engineering specs that this is
happening for every single
concept in human history. Every
word, every image, every
relationship. We are copying
the same static concepts over
and over again.
There's an analogy in the
source material about a city
map that I think really drives
this home, but I want to make
sure I'm picturing it correctly
.
Yes. The city map analogy is
vital for understanding the
structural friction this causes
. Imagine a major metropolitan
city. Now imagine if there was
no single authoritative map of
the streets.
Okay.
Instead, every single delivery
app on your phone had to hire
surveyors to draw their own map
from scratch. Every emergency
service dispatch center has
their own hand-drawn map. Every
individual citizen has to
sketch out their own map to get
to work.
It would be absolute chaos. I
mean, the delivery driver's map
wouldn't show a new detour, the
ambulance map might have a
street name spelled wrong, and
the city planner is working off
a blueprint from 10 years ago.
Exactly. You would have
conflicting addresses, dead
ends on one map that don't
exist on another. And think
about the updating process. If
the city builds a new bridge,
updating the city layout would
be agonizingly slow and error-
prone.
Because everyone has to update
their own copy.
Yes. Every single one of those
thousands of entities has to
independently realize the
bridge was built and then
update their own distinct
static map.
So that is the state of digital
knowledge representation today.
That is what our servers are
doing.
Yes. We are updating millions
of isolated maps instead of
just pointing to the actual
bridge. And every copy can
drift, break, or become stale.
Which perfectly explains why I
get so frustrated when an AI
hallucinates or gives me
outdated information. This
leads us directly into the trap
of the large language models,
the LLMs that are dominating
the headlines right now.
Right.
The white papers we are looking
at are highly critical of the
current LLM paradigm. They use
a very specific phrase. The
cost of digital amnesia. What's
happening there?
What's fascinating here is how
the sources deconstruct the
fundamental limitations of
large language models.
According to the K3D
architectural white papers, LL
Ms try to stuff all of human
knowledge directly into their
internal weights. This is
described as a high dimension,
low density approach.
Okay. Another technical term,
internal weights. What are we
actually talking about when we
say knowledge is stuffed into
weights? Think of the neural
network of an AI, like a
massive complex web of
connections similar to synapses
in a human brain. The weights
are just the mathematical
strength of the connections
between different nodes.
Okay. When a tech company
trains an AI, they feed it pet
abytes of text, and the AI
adjusts these billions of
connection strengths, the
weights, to try and internalize
the patterns of the text.
So instead of having a clean
database where it can just look
up a fact, it's trying to organ
ically memorize the entire
internet by tweaking billions
of little numerical dials.
Precisely. And because it is
trying to memorize everything
within its parameter set, it
suffers from digital amnesia.
What machine learning engineers
often call catastrophic
forgetting.
Catastrophic forgetting.
Yes. As the model learns new
information and adjusts those
dials to accommodate new facts,
it invariably overwrites or
distorts the dials that
represented older information.
It forgets old things as it
learns new things.
And the sources mention this is
restricted by a linear context
window.
I hear that term a lot when
people complain about chatbots
forgetting what was said
earlier in the conversation.
Yes. A context window is
essentially the model's active
working memory.
Because the model processes
text sequentially, word by word
, in a linear feed, it can only
hold so much information in its
active memory at any given
second.
It fills up.
Right. If you feed it a massive
textbook, by the time it gets
to the last chapter, the linear
nature of its processing means
it has largely lost the rigid,
specific context of the first
chapter.
That sounds incredibly
inefficient. And the papers
argue that it's computationally
unsustainable, right?
Yeah.
Because what are the big tech
companies doing to solve this?
They aren't fixing the
architecture. They're just
building bigger models.
Exactly. The current orthodox
solution to digital amnesia and
small context windows is brute
force.
Every single major tech company
is out there independently
scraping and downloading the
entirety of Wikipedia,
thousands of textbooks, endless
Reddit threads, and GitHub
documentation.
Just to jam it all in.
Just to train massive models
that have 175 billion
parameters or even trillions of
parameters.
So going back to the city map
analogy, we have five or six
massive tech giants, each
spending billions of dollars to
send their own surveyors out to
map the exact same digital city
, scraping the exact same
Wikipedia articles to build
their own massive isolated maps
.
They're duplicating the exact
same knowledge.
And they are building isolated
black boxes. That isolation is
key to understanding the PMKR
critique. Because the knowledge
is locked inside those billions
of mathematical weights, it is
completely opaque. It's a black
box.
And you can't just open it up
and read it.
No, you cannot easily inspect
it. You cannot easily audit it.
If an LLM tells a user a
harmful lie or gets a critical
medical fact wrong, an engineer
cannot just go into the
database, find the single fa
ulty entry and correct it.
Because it's not a database
entry. It's a subtle
mathematical relationship
spread across billions of
weights.
Exactly. You cannot easily
update a single fact without
risking damage to the rest of
the model's performance.
The duplication is staggering,
but the opacity is what makes
it a crisis of trust.
How can we build global
infrastructure on a system we
cannot audit or directly edit?
But how does that translate to
the physical world? Because
earlier we talked about the
environmental cost.
This isn't just the theoretical
computer science problem
happening in the cloud.
No, not at all.
The duplication of storage, the
scraping, and the repeated
brute force computational
processing required to train
and run these massive black
boxes has to have a physical
footprint.
The environmental baseline
provided in the sources is
sobering. When you look at the
W3C proposal, the math they use
shows that the current
trajectory of AI compute demand
is fundamentally at odds with
global climate goals.
We are trying to fight climate
change. We are trying to
transition to renewable energy
grids. Yet we are building
digital infrastructures that
require exponentially more
power.
And the tragic irony, according
to these sources, is that this
power isn't being used to
generate new breakthroughs. It
's largely being used to do the
same work, storing the exact
same data over and over again
across different corporate
platforms.
Every time an LLM regenerates a
response based on duplicated
knowledge, every time it halluc
inates and has to be prompted
again, it is burning energy.
Precisely. It is a system built
on redundant friction.
Okay, so we have laid out the
crisis. The problem is massive
duplication, opaque AI black
boxes that we can't audit, and
terrible environmental waste
driven by brute force
computation.
So how do we get out of this
mess?
This is the turning point.
Here's where it gets really
interesting, because the Ecos
ystems AI Studios white papers
introduced the solution.
Procedural memory and the dual
client contract. Let's start
with procedural memory, or PMKR
.
What exactly is a procedural
node and how does it fix the
duplication?
The core fix proposed by PMKR
is a fundamental paradigm shift
in how we define data itself.
Instead of storing knowledge as
heavy static payloads, those
millions of isolated copies of
the letter A across font files,
Unicode, and AI weights,
we store knowledge exactly once
as executable programs and
metadata.
Executable programs. So the
data actually does something.
Yes. The sources call these
procedural nodes.
You store the mathematical
procedure for generating the
knowledge rather than the
finished, pre-rendered output.
There's an analogy in the
sources about baking a cake
that I think is brilliant, but
I want to make sure I fully
grasp the implications.
Can you explain the cake recipe
concept?
Certainly. Think about how the
internet currently works.
Right now, if a million people
want to look at a cake online,
the current digital system
stores a million heavy high-res
olution photographs of that Coke
on servers all over the world.
That is static data. It takes
up a massive amount of hard
drive space, and transferring
those heavy photos across the
network takes significant
bandwidth.
Right. My phone has to download
the heavy image file every time
I load the page.
Exactly. PMKR says, stop
storing the millions of photos.
Instead, store the recipe.
The recipe is the procedural
data. It's just a set of
instructions. Take flour, take
sugar, bake at 350 degrees.
It's just text, basically.
Yes. A recipe takes up almost
no space at all. It is
incredibly lightweight.
And when the user actually
needs to see the cake or
interact with it, the system
executes the recipe and renders
the cake dynamically on the
spot locally on their device.
I have to stop you there and
play devil's advocate for a
second.
If I have to bake the cake
every single time someone wants
to look at the letter A on
their phone, doesn't that take
more computing power than just
loading a static image?
It's a fair point.
I mean, how does executing a
program every time save energy?
Doesn't my phone's processor
have to work harder?
That is the most common and
most important skeptical
question.
It seems counterintuitive.
But Daniel Tempos Ramos points
out in his engineering
specifications that the baking
process is optimized at the
hardware level, specifically
the GPU.
It operates at sub 100 microse
cond latency.
Microseconds. So less than a
fraction of a blink of an eye.
Vastly less.
The reality of modern computing
is that data transfer fetching
a heavy static image from a
server in Virginia, routing it
through undersea cables and
sending it to your phone in
London takes exponentially more
time and energy than it does
for your phone's internal
processor to simply run a tiny
mathematical equation to draw
the image locally.
Ah, I see. So the bottleneck
isn't the processing power. It
's the network transfer and the
massive storage farms required
to hold all those static files.
Precisely. In technical terms,
the PMKR standard uses what
they call symlink-style
composition.
For those of us who aren't
systems administrators, what is
a symlink?
Just like a symbolic link or
shortcut on your computer's
desktop points to a single
actual file buried deep in your
folders, rather than creating a
duplicate of the whole file on
your desktop, PMKR creates
lightweight references.
Oh, yeah, that makes sense.
You have one canonical
procedural source for a concept
, let's say a specific chair.
Every other virtual environment
, video game, or AI system that
needs that chair simply
references that single core
procedure.
So no more duplication, just a
web of pointers looking at one
single source of truth.
Exactly. And this leads to what
the architects call the dual
client contract, which is
perhaps the most philosoph
ically profound technical
innovation in the entire K3D
architecture.
The principle they state in the
manifesto is, "Dual client, one
reality." What does that mean
in practice?
To understand it, we have to
look at the disconnect in
traditional human-AI
interaction loops. Right now,
the human and the AI inhabit
entirely different ontological
worlds.
Ontological worlds, meaning the
fundamental nature of their
reality is different.
Yes. When a human looks at a
computer screen, they see a
visual user interface. They see
buttons, formatted text,
colorful images, layout. But
when an AI looks at that same
system, it doesn't see the
screen.
It just sees code.
It sees an isolated, hidden,
tensor array of mathematical
weights, API endpoints, and raw
code. They are looking at two
completely different
representations of data. The
human is experiencing a visual
illusion, while the AI is
reading a matrix of numbers.
So they aren't actually looking
at the same thing at all. No
wonder the AI sometimes doesn't
understand the context of what
a human is asking about on a
screen.
Precisely. But under the dual
client contract, K3D fixes this
by giving both the human and
the AI the exact same
procedural root source. Let's
go back to our deep dive
example of the character A. In
this new PMKR system, the
character A is stored exactly
once as a procedural node.
Okay, so we have one
mathematical recipe for the
letter A. How does the dual
client aspect work?
When a human client, a person
holding a smartphone, accesses
that node, the system uses the
procedural instructions to
render it in a way humans can
perceive. It executes a
mathematical procedure, like Bé
zier curves, to draw the smooth
visual shape of the letter on a
screen at any resolution.
Wait, before we move on, Bézier
curves. That's a term that pops
up a lot in designs. Instead of
a grid of pixels, it's math.
Exactly. A static image is a
grid of pixels. If you zoom in,
it gets blocky and blurry. A Bé
zier curve is a mathematical
equation that defines a smooth
curve between two points.
Because it's math, you can zoom
in infinitely and the computer
just recalculates the equation
so it stays perfectly crisp. It
's procedural.
Okay, so the human device runs
the equation and shows a
perfectly crisp letter A. Yeah.
Or, as we mentioned earlier, it
could run a different part of
the procedure and render it
tactilely for a Braille reader.
That's the human side. What
about the AI?
When the AI client accesses
that exact same node at the
exact same spatial coordinate
in the system, it doesn't need
to look at the visual Bézier
curves. That would be a waste
of its processing power.
Right.
Instead, it reads the raw
semantic vector embeddings and
the procedural logic that are
directly bound to that
identical geometric node.
The sources use an
architectural blueprint analogy
here, don't they?
Yes. It's a brilliant way to
visualize the dual nature of
this system. Imagine an
architectural blueprint of a
large complex building. An
architect looks at that
blueprint and sees aesthetics.
They see human flow, natural
light, spatial harmony, and
design.
That's the human client viewing
the data.
Exactly. Now, a structural
engineer looks at the exact
same blueprint. But they don't
care about the aesthetic light
flow. They see load-bearing
metrics, material tensile
strengths, stress points, and
sheer walls.
And that's the AI viewing the
data.
Yes. They have entirely
different perceptions based on
their specific needs, but, and
this is the crucial part, they
are looking at one unambiguous
truth. There is no
contradiction between the
architect's vision and the
engineer's math because the
source material is identical.
If you move a wall to change
the light, the load-bearing
metrics update instantly and
automatically.
The white papers refer to this
using mapping terminology. They
say the human gets the rich,
high-resolution aesthetic
texture, what they call UV Map
Zero, and the AI gets a highly
compressed semantic data
texture UV Map One.
For listeners familiar with 3D
modeling, a UV map is basically
how you take a flat 2D image
and wrap it around a 3D object,
like wrapping paper on a box.
Right.
In K3D, UV Map Zero is the
human skin. The human sees the
wood grain on the virtual desk.
UV Map One is the AI skin. The
AI doesn't see wood grain, it
sees a semantic tag that says,
"flammable, structural support,
weight, 50 kilograms." But both
skins are wrapped around the
exact same 3D object in the
exact same location.
And this all takes place in a 3
D spatial architecture. The
sources describe K3D as a
Minecraft for cognition. Which
sounds wild. What does that
actually mean?
That is a very deliberate
metaphor chosen by the
architects. Instead of
knowledge being trapped inside
the invisible abstract neural
weights of a black box model, K
3D externalizes memory into a
persistent navigable 3D
environment. They have specific
terms for this architecture.
They call the persistent long-
term memory the house and the
active working memory the
galaxy.
So it's a literal virtual space
. Yes. For the AI navigating
this 3D space is the equivalent
of traversing meaning. It
physically or virtually rather
moves from concept to concept.
This brings up a huge axiom
stated in the sources one that
blew my mind. They state
spatial proximity equals
semantic similarity. Can you
unpack that?
In current AI, concepts that
are related are mathematically
close together in an invisible,
high-dimensional space that
humans can't see or comprehend.
In K3D, they translate that
math into literal physical
space.
Okay. If two concepts are
closely related in meaning, say
, the concept of a dog and the
concept of a wolf, they are
physically, spatially close to
each other in this 3D virtual
universe.
So if I'm walking through the
house, the room with the dogs
is right next to the room with
the wolves.
Exactly. And this provides what
the machine learning industry
desperately needs right now:
ultimate explainable AI or XAI.
Explainable AI. Because right
now when an LLM gives you an
answer, you have no idea how it
got there. It's just magic math
.
Right. It's a black box. But
because the human and the AI
are in the same shared spatial
reality under PMKR, the human
can put on a VR headset or
simply look at a 2D screen and
literally navigate the 3D space
to see exactly where the AI's
avatar is standing.
You can physically watch the AI
, I think.
Yes. You can see what book of
knowledge its avatar is looking
at, what door it is opening,
what path it took to get from
point A to point B. It ends the
era of the opaque black box
because the AI's reasoning
process becomes a visible, aud
itable journey through a spatial
environment.
That's incredible.
If the AI makes a mistake, you
can walk to where it is
standing and see, oh, it's
looking at the wrong procedural
node. Let's fix that node.
It's like being able to walk
through the physical brain of
the machine while it's thinking
alongside it. That is
absolutely incredible.
Okay. So we've covered the
problem of duplication and
opacity and we've explored the
procedural 3D spatial solution
of the dual client contract.
But what does this actually
mean for the real world?
This is where the numbers come
in.
Because earlier you mentioned
the environmental cost. This
brings us to the carbon impact.
The Ecosystems AI Studios
documents feature a very
detailed K3D carbon blueprint.
And the numbers here are almost
hard to believe.
They are massive. And we should
state clearly for you, the
listener, that we are imparting
the data from these scenario
models exactly as the sources
present them. We are acting as
impartial reporters of this
blueprint modeled by Milton P
once.
The projections are staggering.
They state that if the global
digital infrastructure
transitions to this procedural
architecture, it could result
in 12 gigatons of cumulative
carbon dioxide savings between
the years 2026 and 2035.
12 gigatons. I mean, that
number is so big it almost
loses its meaning. How do they
even arrive at that? What is
the baseline they're comparing
it to?
The baseline is the current
trajectory of data center
growth. AI compute demand is
currently doubling at an
alarming rate. We are building
more and more massive server
farms just to house the dupl
icated data and run the brute
force LLMs. The 12 gigaton
figure represents the image and
avoided by intercepting that
curve with a fundamentally more
efficient architecture.
The blueprint breaks it down
further, projecting that by the
year 2035, this architecture
could save 2.5 gigatons of CO2
equivalent per year. According
to the scenario context
provided in the papers, that
equates to roughly 6.9% of
total global emissions.
To put that 2.5 gigatons per
year into perspective based on
the source comparisons, that is
the equivalent of removing 550
million gasoline powered cars
from the road every single year
.
That's astonishing.
Alternatively, it is equivalent
to the carbon absorption of
planting 21 billion trees. Or
looking at industrial sectors,
saving 2.5 gigatons of CO2 is
roughly double the emissions of
the entire global aviation
industry.
Double aviation. Wow. Imagine
grounding every single flight
on Earth twice over. That is
the scale of the energy waste
we are currently dealing with
in digital duplication and
inefficient AI.
It's a massive claim. So how
exactly do Milton Ponson's
models argue this is achieved?
What are the mechanical drivers
of these savings? The sources
point to two main pillars,
procedural compression and tiny
models. Let's talk about the
compression first.
The procedural memory model
inherently drives unprecedented
data compression. Because you
are replacing millions of heavy
, duplicated static payloads
with single canonical
mathematical procedures and
lightweight symlink references,
the storage footprint drops
dramatically.
They cite a specific benchmark
regarding their character
galaxy. Yes. They ran a test
modeling the data payload
required to represent all the
characters in a system. Under a
traditional payload model
storing the visual pixel arrays
and the data payload.
Under a traditional pixel
arrays and the semantic data
separately, it required 87.7
megabytes. Which doesn't sound
like a lot for one device, but
multiply that by billions of
devices and servers. Exactly.
But by transitioning to PMKR,
using procedural codecs to
replace those static pixel
arrays, that exact same payload
dropped to 26.3 megabytes. Wow.
That is a compression rate of
over 70% achieved with zero
loss of semantic or visual
fidelity.
70% less data to store on hard
drive. 70% less data to push
through network cables. 70%
less data to cool in server
farms. But the really wild part
of the carbon savings, the part
that completely upends the
current AI industry narrative,
comes from how it changes AI
reasoning.
A tiny model.
A tiny model. Right. We talked
earlier about traditional large
language models using 175
billion parameters, or even tr
illions. The K3D architecture
uses something entirely
different. They call them tiny
recursive models, or TRMs.
Yes. And these TRMs operate
with just seven million
parameters. Seven million.
Compared to 175 billion. Wait,
hold on. I have to ask the
obvious question. If it's a
seven million parameter model,
isn't it just, you know, kind
of stupid? How could a model
that small possibly compete
with the reasoning power of
something like GPT-4's massive
parameter count?
It is a totally valid question,
and it represents a paradigm
shift from what is currently
orthodox in machine learning.
To understand it, we have to
look at how traditional LLMs
think. The sources describe
traditional LLMs as utilizing
System 1 thinking.
Borrowing from Daniel Kahneman
's psychology terms, fast,
intuitive thinking.
Exactly. An LLM relies on a
massive static network of
weights doing a single,
incredibly complex forward pass
. It reads the prompt, the math
cascades through the billions
of parameters in one sweep, and
it spits out the most
statistically likely next word.
Okay.
It requires billions of
parameters to hold all that
intuition because it only gets
one shot at the answer per pass
.
But the TRM uses System 2 slow
thinking logic.
Precisely. By utilizing
something called an atomic
procedural RPN stack, a tiny
model can loop recursively.
Okay, you mentioned an RPN
stack, reverse Polish notation.
I haven't heard that term since
my college roommate was
studying LL-level assembly
language. For someone who doesn
't code, what does that actually
mean for an AI?
Think about how we normally
write math: 3 plus 4. The
operator's in the middle. In
reverse Polish notation, you
put the numbers first, then the
operator. 3, 4 plus.
Like an old HP calculator.
Exactly. It seems backward to
humans, but for a computer, it
is incredibly efficient because
it doesn't need to build
complex parsing trees in its
memory to figure out the order
of operations. It just pushes 3
onto a stack, pushes 4 onto a
stack, sees plus, and combines
them.
So how does that apply to an AI
reasoning?
In K3D, the AI uses this RPN
stack for logic execution. It
pushes an intermediate concept
onto the stack, pushes another
concept, applies a relational
operator, evaluates the result,
and loops again. It pushes,
pops, refines, and loops.
That's iterating.
The complexity of the AI's
thought process is derived from
the depth of the recursion, how
many times it loops and refines
, not the static size of the
model's weights.
Ah, so instead of having a
massive brain that reacts
instantly, it has a tiny,
highly efficient brain that is
allowed to sit there and think
through a problem step by step
before answering.
Exactly. Therefore, a 7 million
parameter model looping recurs
ively through a structured 3D
knowledge space can achieve the
targeted reasoning depth of a
massive 70 billion parameter
model. In fact, in some of
their validation tests, they
achieved successful reasoning
loops with models containing
only 2.1 million parameters.
I want to spend some time on
the hardware efficiency of this
because the numbers in the
white paper are just wild. Yeah
.
Think about the physical
computers required for AI today
. If you are a developer and you
want to run a standard,
relatively small, 7 billion
parameter open source model
locally on your machine, you
need about 28 gigabytes of VRAM
, video RAM on your graphics
card.
And 28 gigabytes of VRAM is not
standard. That requires a very
expensive top tier consumer
graphics card or specialized
workstation hardware. It won't
fit on a normal laptop.
Right. But because the K3D TRM
only has 7 million parameters,
its VRAM footprint is infinites
imally small. The sources state
it only requires 8.4 megabytes
of VRAM.
Megabytes, not gigabytes.
From 28 gigabytes down to 8.4
megabytes. The implications for
hardware scaling are profound.
The current AI paradigm dict
ates that we need massive,
multi-million dollar data
centers with specialized
cooling towers just to run
inference APIs. But the sources
state that because the TRM is
so light, you can fit 128
parallel instances of this AI
on a single standard 8 gigabyte
consumer GPU.
128 distinct AI agents running
simultaneously on hardware you
could buy at a local
electronics store.
They ran a benchmark test to
prove this. They had a system
process 500 complex reasoning
questions. The old sequential
way running of the AI.
The old sequential way running
a traditional model took about
50 minutes to grind through
them. But the K3D batched way
running 128 TRM instances in
parallel on a single consumer
card process all 500 questions
in about 24 seconds.
50 minutes down to 24 seconds.
That is 125 times faster.
No cloud dependencies. No
massive data center API calls.
Just cyber and localized bla
zingly fast execution. When you
look at those numbers, 70% data
compression, TRMs using
fractions of a megabyte of VRAM
, 125 X processing speed, it's
no wonder the carbon blueprint
projects such massive
reductions.
Yeah, adds up quickly. If this
compute model displaces even a
fraction of the repetitive AI
tasks currently running in
massive data centers, the
energy grid savings are
historic.
We connect this to the bigger
picture. It explains exactly
why Ecosystems AI Studios isn't
trying to hoard this tech. It
explains why this is being
pushed as a global open
standard. This brings us to the
W3C collaboration. The creators
of PMKR are very explicit in
their manifestos that this is
not meant to be a proprietary
walled garden product.
Right. They aren't trying to be
the next closed open AI or
Google ecosystem. They're
pushing this entire
architecture through the
worldwide web consortium, the W
3C. For those who might not be
familiar with internet
governance, the W3C are
essentially the people who make
the foundational rules for the
web.
The architects of the internet,
basically. Right. They define
HTML, CSS, the protocols that
make the internet universally
accessible. And Ecosystems
wants PMKR to be the open
standard for how the internet
stores and executes knowledge.
The cross community group
momentum documented in the
sources is highly strategic and
very impressive. The PMKR
proposal isn't existing in a
vacuum. It is designed to
integrate seamlessly with
existing major W3C efforts.
Who are they working with?
They highlight deep synergies
with the WebML group, which
involves heavy hitting
stakeholders like Intel,
focusing on standardizing how
machine learning runs
efficiently directly in the web
browser. They are collaborating
with the GPU for the web group,
which involves heavy hitting
stakeholders like Intel,
focusing on standardizing how
machine learning runs
efficiently directly in the web
browser.
They are collaborating with the
GPU for the web group, which
involves Mozilla and Google,
ensuring that those procedural
execution paths we talked about
, the hardware level baking of
the cake, are perfectly
optimized across all devices.
OK, that makes sense.
They are also tightly aligned
with the sustainable web
interest group.
The sources make a crucial
distinction here, too,
regarding older web standards.
The PMKR standard is about
complementing existing web
standards, not ripping them out
and replacing them. They
explicitly state it works
alongside declarative standards
like RDF or JSON-LD.
Let's define those for the
listener. RDF and JSON-LD are
existing methods for linking
data on the web. Think of them
like incredibly detailed name
tags. They are declarative.
They state what a piece of data
is and how it relates to other
data. But they are passive.
They don't actually do anything
.
They describe the data, but
they aren't the engine.
Exactly. PMKR adds an execut
able procedural layer underneath
those name tags. It takes the
descriptive graph of RDF and
gives it executable procedural
continuity. Now, the knowledge
can actually perform operations
efficiently.
And this has massive, tangible
implications for specific
industries. For instance, if
you are a listener working in
the hardware or display
manufacturing space, the W3C PM
KR documents outline a concept
that sounds like science
fiction but is suddenly
mathematically viable:
procedural displays.
This is a fascinating
application of the theory. Let
's unpack procedural displays.
Right now, your phone, your
smartwatch, or your e-reader
renders text by looking up a
pictobased font file stored in
its memory.
Right. The static payload we
talked about in the beginning.
What changes under PMKR?
Because the rendering of the
characters becomes a
mathematical procedural
execution rather than a pixel
array lookup, displays no
longer need to carry heavy dupl
icated font files for every
single language bundle in their
operating system memory.
So imagine a Kindle that doesn
't need to download the English
fonts, the Japanese fonts, the
Arabic fonts. It just contains
the universal mathematical
concept of how to draw human
characters.
Precisely. The device's
hardware executes the procedure
to draw the character perfectly
at any scale. The W3C Web Fonts
Working Group, led by Chris L
illey, is heavily involved in
exploring this. This procedural
rendering enables infinite zoom
without any pixelation. It
enables true, multimodal
accessibility built directly
into the display layer.
Meaning the exact same
procedure that draws the visual
letter on the screen can
simultaneously output the bra
ille equivalent to a haptic
device without needing a
separate translation software
running in the background.
Yes. And it involves
researchers like Ada Rose
Cannon at Samsung looking at
how this radically reduces the
memory footprint of the device
's operating system. It
represents near zero duplic
ation across the entire hardware
ecosystem.
It's an elegant solution to a
massive problem. And frankly,
it's not just the technology
that is fascinating here. It's
the people and the geopolitical
structure behind it. Which
brings us to a historic aspect
of this proposal. The Mercosur
EU partnership.
This is a vital and very
deliberate part of the
narrative presented in the
press kit and the W3C documents
. Historically massive leaps in
computing architecture, the
protocols that define the
digital age have been tightly
controlled by a handful of
massive corporations in a
single geographic region,
usually Silicon Valley.
Right. The typical walled
garden tech monopoly. But the
PMKR community group is driven
by a transatlantic historic
collaboration. The two co-ch
airs of the group represent a
unique partnership. We have
Daniel Campos Ramos, an
electrical engineer and the
primary architect who built the
K3D reference implementation
operating out of Brazil.
Representing the Mercosur
economic block in South America
.
Yes. And he is partnered with
Milton Ponsen, a mathematician
handling the complex
environmental modeling, the
rate distortion theory that
proves the 70% compression, and
the theoretical framing
operating out of the
Netherlands in the European
Union.
The sources emphasize that this
is the first truly
groundbreaking Mercosur EU
joint effort in frontier
technologies. We are seeing raw
functional South American
engineering implementation,
pairing directly with rigorous
European math and the European
technology.
With rigorous European
mathematical and environmental
accountability on a global
stage.
This raises an important
question about the future of
global technical governance.
This W3C initiative proves that
you can build open tech
governance across different
regions, balancing different
institutional contexts and
priorities outside of the
traditional Silicon Valley
monopoly.
And they're doing it entirely
in the open. The philosophy
quoted directly in the manifest
os is, "We patent nothing, we
publish everything, we build in
the open."
The mathematical specifications
and the architectural bluep
rints are published under a
Creative Commons CCBY 4.0
license. And the actual code
contributions for the reference
system are under an Apache 2.0
open source license.
It is a deliberate structural
choice to ensure that the
foundational layer of next
generation digital memory
remains a shared human heritage
, rather than a proprietary
corporate asset locked behind
an expensive API paywall.
The mathematical rigor from the
Netherlands ensures the
compression claims are not just
heuristic observations but
mathematical necessities, while
the engineering from Brazil
proves it can actually run at
sub 100 microsecond latency on
cheap consumer hardware.
Okay, we have covered an
incredible amount of ground
today. Let's summarize the
journey we've just taken
through this massive stack of
sources. We started by looking
at the stark reality of the
knowledge duplication crisis.
We saw how storing data as
static payloads, millions of
isolated copies of the same
font files, the same unicode,
the same semantic vectors, is
creating a fragmented, fragile
internet.
And we examined the brute force
approach of massive large
language models. We saw how
trying to stuff all of human
knowledge into over 175 billion
opaque mathematical weights
leads to digital amnesia,
hallucinations, and black box
systems that we cannot audit.
And worst of all, we saw how
the computational friction of
this duplication is burning uns
ustainable amounts of energy
threatening global climate
goals.
From there, we explored the
elegance of the proposed
solution, procedural memory, or
PMKR. By storing knowledge
exactly once as an executable
mathematical procedure, storing
the recipe instead of a million
photos of the cake, and
referencing it via assuming
style composition, the
architecture mathematically
achieves a 70% data compression
rate.
We explored the philosophical
and technical shift of the dual
client contract. We saw how PMK
R finally aligns human visual
perception and AI semantic
processing into a single shared
reality. The architect sees the
light flow, the engineer sees
the load bearing metrics, but
they're both looking at the
exact same unambiguous
procedural blueprint in a
shared 3D space.
We saw how that shared 3D
spatial reality, that Minecraft
for cognition enables tiny
recursive models with just 7
million parameters to loop and
process using slow thinking
System 2 logic. This allows
them to achieve the reasoning
depth of a massive 175 billion
parameter behemoth, meaning we
can run 128 AI instances in
parallel on a standard consumer
graphics card, processing data
125 times faster.
And we discussed the W3C carbon
blueprint, walking through
Milton Ponson's model scenario,
that if this highly compressed,
hyper-efficient, open standard
is adopted, it could
potentially intercept the
trajectory of data center
growth and save 12 gigatons of
carbon emissions by 2035.
That's 2.5 gigatons a year,
double the global aviation
industry. And we must remind
you, the listener, why this
specific architecture matters
so deeply to your daily
professional and personal life.
Whether you are a regulator
tasked with auditing AI
decisions, and you desperately
need the traceability of that
dual client 3D space, so you
can see exactly where the AI is
standing, or you're a developer
working on sustainable web
technologies aiming to reduce
digital bloat, or a display
manufacturer looking for the
elegant efficiency of
procedural rendering.
This Mercosur EU W3C effort
shifts the very foundation of
how we interact with and store
the digital world. It proves
that extreme efficiency and
radical transparency can be
engineered directly into the
protocol level of the web.
It is a profoundly optimistic
vision for the future of
technology.
I want to leave you with a
final provocative thought,
inspired directly by the
manifestos included in today's
source material.
For the last few years, the
tech industry has largely told
us a single story: that the
future of artificial
intelligence is an ever-growing
, single, towering, privately
owned black box model,
housed in a multi-billion
dollar data center, guarded by
a few elite corporations.
Right.
But what if true intelligence
isn't about hoarding billions
of parameters in the cloud?
What if the real future of
spatial general intelligence is
an open, explorable, 3D
universe of meaning?
A universe where the AI isn't
an opaque, invisible oracle
handing down answers from a
server farm, but rather an
avatar that you can actually
walk alongside, exploring a
shared reality, tracing its
thoughts step-by-step, building
the next era of the web
together.
It challenges us to demand more
from the systems we use every
day.
It really does.
Thank you for joining us for
this deep dive into procedural
memory knowledge representation
and the knowledge 3D
architecture.
Keep questioning the
architecture around you, keep
looking for the procedures
behind the static payloads, and
we'll see you next time.
