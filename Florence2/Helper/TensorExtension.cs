using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Linq;
using System.Numerics;

namespace Florence2
{
    public static class TensorExtension
    {
        public static DenseTensor<T> Ones<T>(ReadOnlySpan<int> dimensions) where T : INumber<T> => Fill(dimensions, T.One);
        public static DenseTensor<T> Zeros<T>(ReadOnlySpan<int> dimensions) where T : INumber<T> => Fill(dimensions, T.Zero);
 

        public static DenseTensor<T> Fill<T>(ReadOnlySpan<int> dimensions, T value) where T : INumber<T>
        {
            var result = new DenseTensor<T>(dimensions);
            result.Fill(value);
            return result;
        }


        public static DenseTensor<float> JoinBatches(params DenseTensor<float>[] tensors)
        {
            if (tensors.Length == 1)
            {
                var tensor = tensors[0];
                var dims = tensors[0].Dimensions.ToArray().Prepend(1).ToArray();
                return new DenseTensor<float>(tensor.Buffer, dims);
            }

            var dimensions = tensors[0].Dimensions.ToArray().Prepend(tensors.Length).ToArray();

            var buffer = new DenseTensor<float>(dimensions);

            for (int i = 0; i < tensors.Length; i++)
            {
                tensors[i].Buffer.CopyTo(buffer.Buffer.Slice(i * (int)tensors[i].Length, (int)tensors[i].Length));
            }

            return buffer;
        }


        /// <summary>
        /// Concatenates the specified tensors along the specified axis.
        /// </summary>
        /// <param name="tensor1">The tensor1.</param>
        /// <param name="tensor2">The tensor2.</param>
        /// <param name="axis">The axis.</param>
        /// <returns></returns>
        /// <exception cref="System.NotImplementedException">Only axis 0,1,2 is supported</exception>
        public static DenseTensor<T> ConcatTensor<T>(this DenseTensor<T> tensor1, DenseTensor<T> tensor2, int axis = 0)
        {
            if (tensor1 == null)
                return tensor2.ToDenseTensor();

            return axis switch
            {
                0 => ConcatenateAxis0(tensor1, tensor2),
                1 => ConcatenateAxis1(tensor1, tensor2),
                2 => ConcatenateAxis2(tensor1, tensor2),
                _ => throw new NotImplementedException("Only axis 0, 1, 2 is supported")
            };
        }


        private static DenseTensor<T> ConcatenateAxis0<T>(this DenseTensor<T> tensor1, DenseTensor<T> tensor2)
        {
            var dimensions = tensor1.Dimensions.ToArray();
            dimensions[0] += tensor2.Dimensions[0];

            var buffer = new DenseTensor<T>(dimensions);
            tensor1.Buffer.CopyTo(buffer.Buffer[..(int)tensor1.Length]);
            tensor2.Buffer.CopyTo(buffer.Buffer[(int)tensor1.Length..]);
            return buffer;
        }


        private static DenseTensor<T> ConcatenateAxis1<T>(DenseTensor<T> tensor1, DenseTensor<T> tensor2)
        {
            var dimensions = tensor1.Dimensions.ToArray();
            dimensions[1] += tensor2.Dimensions[1];
            var concatenatedTensor = new DenseTensor<T>(dimensions);

            if (tensor1.Dimensions.Length == 2)
            {
                for (int i = 0; i < tensor1.Dimensions[0]; i++)
                    for (int j = 0; j < tensor1.Dimensions[1]; j++)
                        concatenatedTensor[i, j] = tensor1[i, j];

                for (int i = 0; i < tensor1.Dimensions[0]; i++)
                    for (int j = 0; j < tensor2.Dimensions[1]; j++)
                        concatenatedTensor[i, j + tensor1.Dimensions[1]] = tensor2[i, j];
            }
            else if (tensor1.Dimensions.Length == 3)
            {
                for (int i = 0; i < tensor1.Dimensions[0]; i++)
                    for (int j = 0; j < tensor1.Dimensions[1]; j++)
                        for (int k = 0; k < tensor1.Dimensions[2]; k++)
                            concatenatedTensor[i, j, k] = tensor1[i, j, k];

                for (int i = 0; i < tensor2.Dimensions[0]; i++)
                    for (int j = 0; j < tensor2.Dimensions[1]; j++)
                        for (int k = 0; k < tensor2.Dimensions[2]; k++)
                            concatenatedTensor[i, j + tensor1.Dimensions[1], k] = tensor2[i, j, k];
            }
            else
            {
                throw new ArgumentException("Length 2 or 3 currently supported");
            }

            return concatenatedTensor;
        }


        private static DenseTensor<T> ConcatenateAxis2<T>(DenseTensor<T> tensor1, DenseTensor<T> tensor2)
        {
            var dimensions = tensor1.Dimensions.ToArray();
            dimensions[2] += tensor2.Dimensions[2];
            var concatenatedTensor = new DenseTensor<T>(dimensions);

            // Copy data from the first tensor
            for (int i = 0; i < dimensions[0]; i++)
                for (int j = 0; j < dimensions[1]; j++)
                    for (int k = 0; k < tensor1.Dimensions[2]; k++)
                        concatenatedTensor[i, j, k] = tensor1[i, j, k];

            // Copy data from the second tensor
            for (int i = 0; i < dimensions[0]; i++)
                for (int j = 0; j < dimensions[1]; j++)
                    for (int k = 0; k < tensor2.Dimensions[2]; k++)
                        concatenatedTensor[i, j, k + tensor1.Dimensions[2]] = tensor2[i, j, k];

            return concatenatedTensor;
        }
    }
}