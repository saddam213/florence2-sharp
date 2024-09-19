using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Model;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace Florence2;

public interface IModelSource
{
    public enum Model
    {
        DecoderModelMerged,
        EmbedTokens,
        EncoderModel,
        VisionEncoder
    }
    public bool TryGetModelPath(IModelSource.Model model, out string modelPath);

    public byte[] GetModelBytes(IModelSource.Model model);

}
public class Florence2Model
{
    private readonly SessionOptions _sessionOptions;
    private readonly OnnxModelSession _sessionEmbedTokens;
    private readonly OnnxModelSession _sessionVisionEncoder;
    private readonly OnnxModelSession _sessionEncoder;
    private readonly OnnxModelSession _sessionDecoderMerged;
    private readonly Florence2Tokenizer _tokenizer;
    private readonly CLIPImageProcessor _imageProcessor;
    private readonly Florence2PostProcessor _postProcessor;

    private InferenceSession GetSessionForModel(IModelSource source, IModelSource.Model model)
    {
        return source.TryGetModelPath(model, out var modelPath)
            ? new InferenceSession(modelPath, _sessionOptions)
            : new InferenceSession(source.GetModelBytes(model), _sessionOptions);

    }

    private OnnxModelSession GetOnnxStackSessionForModel(IModelSource source, IModelSource.Model model)
    {
        source.TryGetModelPath(model, out var modelPath);
        var onnxConfig = new OnnxModelConfig
        {
            DeviceId = 0,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            ExecutionProvider = ExecutionProvider.Cpu,
            InterOpNumThreads = 0,
            IntraOpNumThreads = 0,
            OnnxModelPath = modelPath
        };
        return new OnnxModelSession(onnxConfig);
    }

    public Florence2Model(IModelSource modelSource, SessionOptions sessionOptions = null)
    {
        _sessionOptions = sessionOptions ?? new SessionOptions();
        _sessionDecoderMerged = GetOnnxStackSessionForModel(modelSource, IModelSource.Model.DecoderModelMerged);
        _sessionEmbedTokens = GetOnnxStackSessionForModel(modelSource, IModelSource.Model.EmbedTokens);
        _sessionEncoder = GetOnnxStackSessionForModel(modelSource, IModelSource.Model.EncoderModel);
        _sessionVisionEncoder = GetOnnxStackSessionForModel(modelSource, IModelSource.Model.VisionEncoder);

        _tokenizer = Florence2Tokenizer.Init();

        _imageProcessor = new CLIPImageProcessor(new CLIPImageProcessor.CLIPConfig()
        {
            ImageMean = [0.485f, 0.456f, 0.406f],
            ImageSeqLength = 577,
            ImageStd = [0.229f, 0.224f, 0.225f],
            RescaleFactor = 0.00392156862745098f,
            CropHeight = 768,
            CropWidth = 768,
        }, KnownResamplers.Bicubic);
        _postProcessor = new Florence2PostProcessor();

    }


    private string ConstructPrompts(TaskTypes taskType, string textInput = null)
    {
        if (TaskPromptsWithoutInputsDict.TryGetValue(taskType, out var taskPrompt))
        {
            return taskPrompt;
        }
        else if (TaskPromptsWithInputDict.TryGetValue(taskType, out var taskPromptFormat))
        {
            if (textInput is null) throw new ArgumentNullException(nameof(textInput), "expected text with this taskType");
            return string.Format(taskPromptFormat, textInput);
        }
        else
        {
            throw new ArgumentException("not found task type" + taskType, nameof(taskType));
        }
    }

    public async Task<FlorenceResults[]> Run(TaskTypes task, Stream imgStream, string textInput, CancellationToken cancellationToken)
    {
        var configuration = new NormalizedConfig();
        var prompts = new string[] { ConstructPrompts(task, textInput) };
        var (inputIdsForEncoder, attentionMaskForEncoder) = GetTextInputs(prompts);
        var (pixelValues, imgSize) = _imageProcessor.Preprocess(imgStream);


        var sessionEmbedMetadata = await _sessionEmbedTokens.GetMetadataAsync();
        using (var sessionEmbedParams = new OnnxInferenceParameters(sessionEmbedMetadata))
        {
            sessionEmbedParams.AddInputTensor(inputIdsForEncoder);
            sessionEmbedParams.AddOutputBuffer([inputIdsForEncoder.Dimensions[0], inputIdsForEncoder.Dimensions[1], configuration.DecoderHiddenSize]);
            var sessionEmbedResult = await _sessionEmbedTokens.RunInferenceAsync(sessionEmbedParams);
            using (var inputsEmbeds = sessionEmbedResult.First())
            {
                pixelValues = TensorExtension.JoinBatches(pixelValues);

                var sessionVisionMetadata = await _sessionVisionEncoder.GetMetadataAsync();
                using (var sessionVisionParams = new OnnxInferenceParameters(sessionVisionMetadata))
                {
                    sessionVisionParams.AddInputTensor(pixelValues);
                    sessionVisionParams.AddOutputBuffer();
                    var sessionVisionResult = _sessionVisionEncoder.RunInference(sessionVisionParams);
                    var imageFeatures = sessionVisionResult.First().ToDenseTensor();

                    var (inputsEmbedsMerged, attentionMaskMerged) = MergeInputIdsWithImageFeatures(inputsEmbeds.ToDenseTensor(), imageFeatures, attentionMaskForEncoder);

                    var sessionEncoderMetadata = await _sessionEncoder.GetMetadataAsync();
                    using (var sessionEncoderParams = new OnnxInferenceParameters(sessionEncoderMetadata))
                    {
                        sessionEncoderParams.AddInputTensor(attentionMaskMerged);
                        sessionEncoderParams.AddInputTensor(inputsEmbedsMerged);
                        sessionEncoderParams.AddOutputBuffer(inputsEmbedsMerged.Dimensions);

                        var sessionEncoderResult = await _sessionEncoder.RunInferenceAsync(sessionEncoderParams);
                        using (var lastHiddenState = sessionEncoderResult.First())
                        {
                            var result = await GenerationLoop(configuration, attentionMaskMerged, lastHiddenState.ToDenseTensor());
                            return result.Select(r => _postProcessor.PostProcessGeneration(r, task, imgSize)).ToArray();
                        }
                    }
                }
            }
        }


    }

    private async Task<List<string>> GenerationLoop(NormalizedConfig configuration, DenseTensor<long> attentionMask, DenseTensor<float> encoder_outputs)
    {
        var batchSize = 1;
        var batchIndex = 0;
        var maxLength = GenerationConfig.MaxLength;
        var numBeams = GenerationConfig.NumBeams;
        var topK = GenerationConfig.TopK;

        int noRepeatNgramSize = GenerationConfig.NoRepeatNgramSize;

        var decoderStartTokenID = _tokenizer.TokenToID(_tokenizer.Tokens.EndOfSequence);
        var decoderInputIds = TensorExtension.Fill<long>(new[] { batchSize, 1 }, decoderStartTokenID);
        var allInputIds = Enumerable.Range(0, batchSize).Select(_ => new List<long>([decoderStartTokenID])).ToArray();


        var results = new List<string>();
        var pastKeyValues = default(DenseTensor<float>[]);

        var sampler = new BeamSearchSampler(TensorOperationRegistry.TopKSession(_sessionOptions), topK: topK, numBeams: numBeams);

        var logitsProcessors = new List<LogitsProcessor>
        {
            new NoRepeatNGramLogitsProcessor(noRepeatNgramSize),
            new ForcedBOSTokenLogitsProcessor(_tokenizer.TokenToID(_tokenizer.Tokens.BeginningOfSequence)),
            new ForcedEOSTokenLogitsProcessor(maxLength, _tokenizer.TokenToID(_tokenizer.Tokens.EndOfSequence))
        };

        var stoppingCriteria = new List<StoppingCriteria>
        {
            new MaxLengthCriteria(maxLength),
            new EosTokenCriteria(_tokenizer.TokenToID(_tokenizer.Tokens.EndOfSequence))
        };


        var decoder = new ByteLevelDecoder(_tokenizer.AddedTokens);

        double[] scores = new double[batchSize];

        while (true)
        {
            var sessionEmbedMetadata = await _sessionEmbedTokens.GetMetadataAsync();
            using (var sessionEmbedParams = new OnnxInferenceParameters(sessionEmbedMetadata))
            {
                sessionEmbedParams.AddInputTensor(decoderInputIds);
                sessionEmbedParams.AddOutputBuffer([1, 1, configuration.EncoderHiddenSize]);

                var sessionEmbedResult = await _sessionEmbedTokens.RunInferenceAsync(sessionEmbedParams);
                using (var decoderInputsEmbeds = sessionEmbedResult.First())
                {
                    var useCacheBranche = pastKeyValues is not null;
                    var useCacheBranch = new DenseTensor<bool>(new[] { useCacheBranche }, [1]);

                    var sessionDecoderMergedMetadata = await _sessionDecoderMerged.GetMetadataAsync();
                    using (var sessionDecoderMergedParams = new OnnxInferenceParameters(sessionDecoderMergedMetadata))
                    {
                        sessionDecoderMergedParams.AddInputTensor(attentionMask);
                        sessionDecoderMergedParams.AddInputTensor(encoder_outputs);
                        sessionDecoderMergedParams.AddInput(decoderInputsEmbeds);

                        pastKeyValues ??= InitPastKeyValues(configuration);
                        foreach (var pastKeyValue in pastKeyValues)
                            sessionDecoderMergedParams.AddInputTensor(pastKeyValue);

                        sessionDecoderMergedParams.AddInputTensor(useCacheBranch);

                        foreach (var output in sessionDecoderMergedMetadata.Outputs)
                            sessionDecoderMergedParams.AddOutputBuffer();


                        using (var sessionDecoderMergedResult = _sessionDecoderMerged.RunInference(sessionDecoderMergedParams))
                        {
                            FromPresent(sessionDecoderMergedResult, pastKeyValues, useCacheBranche);

                            var logitsTensor = sessionDecoderMergedResult[0].ToDenseTensor();
                            var logitsTensorProcessed = logitsTensor.Reshape([logitsTensor.Dimensions[0], logitsTensor.Dimensions[2]]).ToDenseTensor();

                            foreach (var logitsProcessor in logitsProcessors)
                            {
                                logitsProcessor.Process(batchIndex, allInputIds[batchIndex].ToArray(), logitsTensorProcessed);
                            }

                            var sampledTokens = sampler.Sample(batchIndex, logitsTensorProcessed);

                            var generatedInputIds = new List<long>[batchSize];


                            foreach (var (token, score) in sampledTokens)
                            {
                                scores[batchIndex] += score;
                                var batchAllInputIds = allInputIds[batchIndex] ?? new List<long>();
                                batchAllInputIds.Add(token);
                                allInputIds[batchIndex] = batchAllInputIds;

                                var batchgeneratedInputIds = generatedInputIds[batchIndex] ?? new List<long>();
                                batchgeneratedInputIds.Add(token);
                                generatedInputIds[batchIndex] = batchgeneratedInputIds;
                                // TODO: Support beam search or just remove this
                                break;
                            }


                            var isDone = new bool[batchSize];

                            foreach (var stoppingCriterion in stoppingCriteria)
                            {
                                var criterionDone = stoppingCriterion.Call(allInputIds, scores);

                                for (var i = 0; i < isDone.Length; ++i)
                                {
                                    isDone[i] = isDone[i] || criterionDone[i];
                                }
                            }

                            if (isDone.All(e => e))
                            {
                                results.AddRange(allInputIds.Select(allInputId => DecodeSingle(_tokenizer, decoder, allInputId.Select(Convert.ToInt32).ToArray())));
                                break;
                            }
                            else
                            {
                                decoderInputIds = new DenseTensor<long>(generatedInputIds.SelectMany(ids => ids).ToArray(), [generatedInputIds.Length, 1]);
                            }
                        }

                    }
                }
            }
        }

        return results;
    }


    private (DenseTensor<long> inputIds, DenseTensor<long> attentionMask) GetTextInputs(string[] sentences)
    {
        var numSentences = sentences.Length;

        var encoded = _tokenizer.Encode(sentences);

        var tokenCount = encoded.First().InputIds.Length;

        var inputIds = new long[encoded.Sum(s => s.InputIds.Length)];
        var flattenAttentionMask = new long[encoded.Sum(s => s.AttentionMask.Length)];

        var flattenInputIDsSpan = inputIds.AsSpan();
        var flattenAttentionMaskSpan = flattenAttentionMask.AsSpan();

        foreach (var (InputIds, AttentionMask) in encoded)
        {
            InputIds.AsSpan().CopyTo(flattenInputIDsSpan);
            flattenInputIDsSpan = flattenInputIDsSpan.Slice(InputIds.Length);

            AttentionMask.AsSpan().CopyTo(flattenAttentionMaskSpan);
            flattenAttentionMaskSpan = flattenAttentionMaskSpan.Slice(AttentionMask.Length);
        }

        var dimensions = new[] { numSentences, tokenCount };

        return (inputIds: new DenseTensor<long>(inputIds, dimensions), attentionMask: new DenseTensor<long>(flattenAttentionMask, dimensions));

    }

    private static (DenseTensor<float> inputs_embeds, DenseTensor<long> attentionMask) MergeInputIdsWithImageFeatures(
        DenseTensor<float> inputsEmbeds,
        DenseTensor<float> imageFeatures,
        DenseTensor<long> attentionMask
    )
    {
        return (
            inputs_embeds: TensorExtension.ConcatTensor(
                imageFeatures, // image embeds
                inputsEmbeds, // task prefix embeds
                axis: 1),
            attentionMask: TensorExtension.ConcatTensor(
                TensorExtension.Ones<long>(imageFeatures.Dimensions.Slice(0, 2)), // image attention mask
                attentionMask, // task prefix attention mask
                axis: 1));
    }

    private static string DecodeSingle(Florence2Tokenizer tokenizer, ByteLevelDecoder decoder, int[] token_ids)
    {
        var tokens = token_ids.Select(tokenizer.IdToToken);

        var decoded = string.Join(string.Empty, decoder.DecodeChain(tokenizer, tokens.ToArray()));
        decoded = CleanUpTokenization(decoded);

        return decoded;
    }

    private static string CleanUpTokenization(string text)
    {
        // Clean up a list of simple English tokenization artifacts
        // like spaces before punctuations and abbreviated forms
        return text.Replace(" .", ".")
           .Replace(" ?", "?")
           .Replace(" !", "!")
           .Replace(" ,", ",")
           .Replace(" ' ", "")
           .Replace(" n't", "n't")
           .Replace(" 'm", "'m")
           .Replace(" 's", "'s")
           .Replace(" 've", "'ve")
           .Replace(" 're", "'re");
    }







    private void FromPresent(IDisposableReadOnlyCollection<OrtValue> decoderOutput, DenseTensor<float>[] pastKeyValues, bool useCache)
    {
        var presentOutputs = decoderOutput.Select(x => x.ToDenseTensor()).ToArray()[1..];
        for (int i = 0; i < presentOutputs.Length; i++)
        {
            if (i % 4 == 0)
            {
                pastKeyValues[i] = presentOutputs[i].ToDenseTensor();
                pastKeyValues[i + 1] = presentOutputs[i + 1].ToDenseTensor();
                if (!useCache)
                {
                    pastKeyValues[i + 2] = presentOutputs[i + 2].ToDenseTensor();
                    pastKeyValues[i + 3] = presentOutputs[i + 3].ToDenseTensor();
                }
            }
        }
    }



    private DenseTensor<float>[] InitPastKeyValues(NormalizedConfig normalizedConfig)
    {
        var batchSize = 1;
        var output = new DenseTensor<float>[normalizedConfig.NumDecoderLayers * 4];

        var encoderDimKv = normalizedConfig.EncoderHiddenSize / normalizedConfig.NumEncoderHeads;
        var decoderDimKv = normalizedConfig.DecoderHiddenSize / normalizedConfig.NumDecoderHeads;

        var encoderDims = new[] { batchSize, normalizedConfig.NumDecoderHeads, 0, encoderDimKv };
        var decoderDims = new[] { batchSize, normalizedConfig.NumDecoderHeads, 0, decoderDimKv };

        for (var i = 0; i < output.Length; ++i)
        {
            if (i % 4 == 0)
            {
                output[i] = new DenseTensor<float>(decoderDims);
                output[i + 1] = new DenseTensor<float>(decoderDims);
                output[i + 2] = new DenseTensor<float>(encoderDims);
                output[i + 3] = new DenseTensor<float>(encoderDims);
            }
        }
        return output;
    }


    public static Dictionary<TaskTypes, string> TaskPromptsWithoutInputsDict = new Dictionary<TaskTypes, string>
    {
        { TaskTypes.OCR, "What is the text in the image?" },
        { TaskTypes.OCR_WITH_REGION, "What is the text in the image, with regions?" },
        { TaskTypes.CAPTION, "What does the image describe?" },
        { TaskTypes.DETAILED_CAPTION, "Describe in detail what is shown in the image." },
        { TaskTypes.MORE_DETAILED_CAPTION, "Describe with a paragraph what is shown in the image." },
        { TaskTypes.OD, "Locate the objects with category name in the image." },
        { TaskTypes.DENSE_REGION_CAPTION, "Locate the objects in the image, with their descriptions." },
        { TaskTypes.REGION_PROPOSAL, "Locate the region proposals in the image." }
    };

    public Dictionary<TaskTypes, string> TaskPromptsWithInputDict = new Dictionary<TaskTypes, string>
    {
        { TaskTypes.CAPTION_TO_PHRASE_GROUNDING, "Locate the phrases in the caption: {0}" },
        { TaskTypes.REFERRING_EXPRESSION_SEGMENTATION, "Locate {0} in the image with mask" },
        { TaskTypes.REGION_TO_SEGMENTATION, "What is the polygon mask of region {0}" },
        { TaskTypes.OPEN_VOCABULARY_DETECTION, "Locate {0} in the image." },
        { TaskTypes.REGION_TO_CATEGORY, "What is the region {0}?" },
        { TaskTypes.REGION_TO_DESCRIPTION, "What does the region {0} describe?" },
        { TaskTypes.REGION_TO_OCR, "What text is in the region {0}?" },
    };
}