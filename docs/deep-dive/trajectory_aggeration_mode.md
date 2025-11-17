
<!-- 标题： Adopting the Trajectory Aggregation Mode for 更快的training -->
# Adopting the Trajectory Aggregation Mode for Faster Training

## Introduction
<!-- intro: 由于transition的模式比较慢，我们需要一个trajectory mode加快运行速度 -->

The transition-based aggregation mode suffers from significant efficiency limitations. To accelerate training speed, we need to implement a trajectory-based aggregation mode that processes multiple turns as a single training sample.

## Implementation Challenges of Trajectory Mode with Response Masking

### 1. Special Token Mismatch
<!-- 1. special token mismatch，我们无法从chat template中还原content，我们发现response中LLM生成的end token可能是<end_of_text>或者<eot_id>，但是在下一轮中作为prompt的message的一部分输入时可能会变 -->
A fundamental challenge arises from the irreversibility of chat templates. We cannot accurately reconstruct the original content from tokenized sequences due to inconsistencies in special token handling. Specifically, the LLM may generate different end tokens (e.g., `<end_of_text>` or `<eot_id>`) during response generation. However, when these generated responses are incorporated into the next turn's prompt as part of the conversation history, the end tokens may be transformed or normalized differently by the chat template.

### 2. Retokenization Inconsistencies

<!-- 2. retokenize的时候的问题：作为response生成的token和string一一对应，但是下一次的prompt会对上一次的string做一次retokenize，同样的string作为prompt和response的时候的token id可能不同 -->
During response generation, each token corresponds one-to-one with the generated string. However, when this string is retokenized as part of the next turn's prompt, the same text sequence may produce different token IDs depending on its context (prompt vs. response).

This manifests in several ways:

**a) Cross-turn boundary tokenization artifacts**
<!-- 1. 如果不是message的话，prompt和response界限的问题。（一个图片的例子，如果response的结尾是感叹号，然后下一次agent的逻辑还是在这条message后面直接拼接\n\n，那么!\n\n可能会被retokenize成一个新的token，其中一半内容来自于上一个turn的response，一半来自下一个turn的prompt） -->
When responses are not structured as separate messages, boundary ambiguities emerge. For example, if a response ends with an exclamation mark "!" and the agent logic appends "\n\n" directly afterward, the sequence "!\n\n" may be retokenized as a single token during the next turn. This creates a token that spans two turns—half from the previous response and half from the next prompt—making it impossible to correctly assign response masks.

**b) Whitespace and escape character discrepancies**
<!-- 2. 作为新的prompt，会出现多出来的空格、转义符等小的区别 -->
When the same text is retokenized as a new prompt, minor differences appear, such as additional whitespace, escape characters, or formatting variations that alter the token sequence.

**c) Reserved token recognition**
<!-- 3. reserved token会被处理成special token，导致mismatch（response生成乱码了，乱码依次生成了<researved special token xx>这些分段的字符，多个token id;但是在retokenize的时候会被识别成一个researved special token，变成了一个token） -->
When the model generates garbled output that accidentally forms a reserved special token character sequence (e.g., generating "`<reserved_token_xx>`" as separate text tokens across multiple token IDs), retokenization may recognize this as an actual special token, collapsing multiple tokens into a single special token ID. This creates a token count mismatch between generation and retokenization.

### 3. Agent Post-processing Modifications

<!-- 3. agent中的后处理，例如truncated一些结果。例如有些agent需要在response结束之后通过正则表达式truncate一部分对后文没用的content，导致我们通过识别prefix没办法对应上（因为前一段的response少了一部分内容 -->
Many agent implementations apply post-processing transformations to generated responses, such as regex-based truncation to remove unnecessary content before the next turn. This creates a mismatch where the previous turn's response in our stored rollout data differs from the actual prompt prefix used in the subsequent turn, preventing accurate prefix matching.


## Consequences

These challenges lead to two critical failures:

### 1. Data Processing Stage Failure

<!-- 1. 在数据处理阶段，prefix和新的prompt之间无法match，不是exactly the same，无法识别成同一个trajectory的多次call -->
During data preparation, the stored response prefix cannot be matched with the new prompt—they are not exactly identical. This prevents the system from correctly identifying multiple calls as belonging to the same trajectory, breaking the trajectory aggregation logic.

### 2. Training Stage Failure

<!-- 2. 训练阶段，response mask可能加的位置不对，导致接头的地方的token加了错误的mask，例如同一个token不知道该作为prompt还是response部分 --> -->
Response masks may be applied to incorrect token positions. At turn boundaries, tokens may receive incorrect mask values, causing ambiguity about whether a token should be treated as part of the prompt (mask=0) or response (mask=1). This corrupts the training signal and degrades model learning.



<!-- 
# 我们的解决方案

1. 使用模糊匹配的方式判断是否可以merge，先在id阶段去容忍一定程度的special token mismatch，然后转换为string，最大程度上保证匹配，而且对于空格、转义符等mismatch设置一定程度的容忍度。我们推荐设置special token为用户当前agent的max turn的数值，string tolerance为message的数值（max turn*2）

2. 对于可以merge的trasition，使用最后的一条的prompt+response作为这条数据的input id和response id，并且

2. 用最开始的prompt作为prompt的长度，用response mask来区分multi-turn的response部分和新增的prompt部分






# 我们希望用户配合的setting：

1. 用message的形式去叠新的request

2. 给正确的max prompt length和max response length

2. 注意agent中的postprocess的逻辑


# 风险：off policy问题

解决方案 -->