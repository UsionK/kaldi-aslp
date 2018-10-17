// nnet/nnet-component-test.cc
// Copyright 2014-2015  Brno University of Technology (author: Karel Vesely),
//                      The Johns Hopkins University (author: Sri Harish Mallidi)
// Copyright 2016  ASLP (Author: zhangbinbin liwenpeng duwei)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <sstream>
#include <fstream>
#include <algorithm>

#include "util/common-utils.h"

#include "aslp-nnet/nnet-component.h"
#include "aslp-nnet/nnet-nnet.h"
#include "aslp-nnet/nnet-convolutional-component.h"
#include "aslp-nnet/nnet-max-pooling-component.h"
#include "aslp-nnet/nnet-fsmn.h"
#include "aslp-nnet/nnet-bi-compact-vfsmn.h"

namespace kaldi {
namespace aslp_nnet {

  /*
   * Helper functions
   */  
  template<typename Real>
  void ReadCuMatrixFromString(const std::string& s, CuMatrix<Real>* m) {
    std::istringstream is(s + "\n");
    m->Read(is, false); // false for ascii
  }

  Component* ReadComponentFromString(const std::string& s) {
    std::istringstream is(s + "\n");
    return Component::Read(is, false); // false for ascii
  }


  /*
   * Unit tests,
   */
  void UnitTestLengthNorm() {
    // make L2-lenght normalization component,
    Component* c = ReadComponentFromString("<LengthNormComponent> 5 5");
    // prepare input,
    CuMatrix<BaseFloat> mat_in;
    ReadCuMatrixFromString("[ 1 2 3 4 5 \n 2 3 5 6 8 ] ", &mat_in);
    // propagate,
    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in,&mat_out);
    // check the lenght,
    mat_out.MulElements(mat_out); // ^2,
    CuVector<BaseFloat> check_length_is_one(2);
    check_length_is_one.AddColSumMat(1.0, mat_out, 0.0); // sum_of_cols(x^2),
    check_length_is_one.ApplyPow(0.5); // L2norm = sqrt(sum_of_cols(x^2)),
    CuVector<BaseFloat> ones(2); ones.Set(1.0);
    AssertEqual(check_length_is_one, ones);
  }

  void UnitTestConvolutionalComponentUnity() {
    // make 'identity' convolutional component,
    Component* c = ReadComponentFromString("<ConvolutionalComponent> 5 5 \
      <PatchDim> 1 <PatchStep> 1 <PatchStride> 5 \
      <LearnRateCoef> 1.0 <BiasLearnRateCoef> 1.0 \
      <MaxNorm> 0 \
      <Filters> [ 1 \
      ] <Bias> [ 0 ]"
    );
    
    // prepare input,
    CuMatrix<BaseFloat> mat_in;
    ReadCuMatrixFromString("[ 1 2 3 4 5 ] ", &mat_in);
    
    // propagate,
    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in,&mat_out);
    KALDI_LOG << "mat_in" << mat_in << "mat_out" << mat_out;
    AssertEqual(mat_in,mat_out);

    // backpropagate,
    CuMatrix<BaseFloat> mat_out_diff(mat_in), mat_in_diff;
    c->Backpropagate(mat_in, mat_out, mat_out_diff, &mat_in_diff);
    KALDI_LOG << "mat_out_diff " << mat_out_diff << " mat_in_diff " << mat_in_diff;
    AssertEqual(mat_out_diff,mat_in_diff);
    
    // clean,
    delete c;
  }

  void UnitTestConvolutionalComponent3x3() {
    // make 3x3 convolutional component, design such weights and input so output is zero,
    Component* c = ReadComponentFromString("<ConvolutionalComponent> 9 15 \
      <PatchDim> 3 <PatchStep> 1 <PatchStride> 5 \
      <LearnRateCoef> 1.0 <BiasLearnRateCoef> 1.0 \
      <MaxNorm> 0 \
      <Filters> [ -1 -2 -7   0 0 0   1 2 7 ; \
                  -1  0  1  -3 0 3  -2 2 0 ; \
                  -4  0  0  -3 0 3   4 0 0 ] \
      <Bias> [ -20 -20 -20 ]"
    );
    
    // prepare input, reference output,
    CuMatrix<BaseFloat> mat_in;
    ReadCuMatrixFromString("[ 1 3 5 7 9  2 4 6 8 10  3 5 7 9 11 ]", &mat_in);
    CuMatrix<BaseFloat> mat_out_ref;
    ReadCuMatrixFromString("[ 0 0 0  0 0 0  0 0 0 ]", &mat_out_ref);
    
    // propagate,
    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in, &mat_out);
    KALDI_LOG << "mat_in" << mat_in << "mat_out" << mat_out;
    AssertEqual(mat_out, mat_out_ref);

    // prepare mat_out_diff, mat_in_diff_ref,
    CuMatrix<BaseFloat> mat_out_diff;
    ReadCuMatrixFromString("[ 1 0 0  1 1 0  1 1 1 ]", &mat_out_diff);
    CuMatrix<BaseFloat> mat_in_diff_ref; // hand-computed back-propagated values,
    ReadCuMatrixFromString("[ -1 -4 -15 -8 -6   0 -3 -6 3 6   1 1 14 11 7 ]", &mat_in_diff_ref);

    // backpropagate,
    CuMatrix<BaseFloat> mat_in_diff;
    c->Backpropagate(mat_in, mat_out, mat_out_diff, &mat_in_diff);
    KALDI_LOG << "mat_in_diff " << mat_in_diff << " mat_in_diff_ref " << mat_in_diff_ref;
    AssertEqual(mat_in_diff, mat_in_diff_ref);
    
    // clean,
    delete c;
  }



  void UnitTestMaxPoolingComponent() {
    // make max-pooling component, assuming 4 conv. neurons, non-overlapping pool of size 3,
    Component* c = Component::Init("<MaxPoolingComponent> <InputDim> 24 <OutputDim> 8 \
                     <PoolSize> 3 <PoolStep> 3 <PoolStride> 4");

    // input matrix,
    CuMatrix<BaseFloat> mat_in;
    ReadCuMatrixFromString("[ 3 8 2 9 \
                              8 3 9 3 \
                              2 4 9 6 \
                              \
                              2 4 2 0 \
                              6 4 9 4 \
                              7 3 0 3;\
                              \
                              5 4 7 8 \
                              3 9 5 6 \
                              3 4 8 9 \
                              \
                              5 4 5 6 \
                              3 1 4 5 \
                              8 2 1 7 ]", &mat_in);

    // expected output (max values in columns),
    CuMatrix<BaseFloat> mat_out_ref;
    ReadCuMatrixFromString("[ 8 8 9 9 \
                              7 4 9 4;\
                              5 9 8 9 \
                              8 4 5 7 ]", &mat_out_ref);
    
    // propagate,
    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in,&mat_out);
    KALDI_LOG << "mat_out" << mat_out << "mat_out_ref" << mat_out_ref;
    AssertEqual(mat_out, mat_out_ref);

    // locations of max values will be shown,
    CuMatrix<BaseFloat> mat_out_diff(mat_out);
    mat_out_diff.Set(1);
    // expected backpropagated values,
    CuMatrix<BaseFloat> mat_in_diff_ref; // hand-computed back-propagated values,
    ReadCuMatrixFromString("[ 0 1 0 1 \
                              1 0 1 0 \
                              0 0 1 0 \
                              \
                              0 1 0 0 \
                              0 1 1 1 \
                              1 0 0 0;\
                              \
                              1 0 0 0 \
                              0 1 0 0 \
                              0 0 1 1 \
                              \
                              0 1 1 0 \
                              0 0 0 0 \
                              1 0 0 1 ]", &mat_in_diff_ref);
    // backpropagate,
    CuMatrix<BaseFloat> mat_in_diff;
    c->Backpropagate(mat_in, mat_out, mat_out_diff, &mat_in_diff);
    KALDI_LOG << "mat_in_diff " << mat_in_diff << " mat_in_diff_ref " << mat_in_diff_ref;
    AssertEqual(mat_in_diff, mat_in_diff_ref);

    delete c;
  }

  void UnitTestFsmn() { /* Implemented by Kaituo XU */
      Component* cp = Component::Init(
              "<Fsmn> <InputDim> 5 <OutputDim> 5 <LOrder> 5 <ROrder> 3 <LStride> 1 <RStride> 1"
              );
      Fsmn* c = dynamic_cast<Fsmn*>(cp);
      Component::ComponentType t = c->GetType();
      KALDI_LOG << c->TypeToMarker(t);
      KALDI_LOG << c->Info();


      CuMatrix<BaseFloat> mat_in;
      ReadCuMatrixFromString("[ 1.   2.   3.   4.   5. \n\
              6.   7.   8.   9.  10. \n\
              11.  12.  13.  14.  15. \n\
              16.  17.  18.  19.  20. \n\
              21.  22.  23.  24.  25. \n\
              26.  27.  28.  29.  30. \n\
              31.  32.  33.  34.  35. \n\
              36.  37.  38.  39.  40. \n\
              41.  42.  43.  44.  45. \n\
              46.  47.  48.  49.  50. \n\
              51.  52.  53.  54.  55. \n\
              56.  57.  58.  59.  60. \n\
              61.  62.  63.  64.  65. \n\
              66.  67.  68.  69.  70. \n\
              71.  72.  73.  74.  75. ]", &mat_in);
      KALDI_LOG << mat_in.NumRows() << " " << mat_in.NumCols();
      KALDI_LOG << mat_in;

      CuMatrix<BaseFloat> tflag;
      // ReadCuMatrixFromString("[ 0 \n 1 \n 2 \n 3 \n 4 \n 5 \n 6 \n 7 \n 0 \n 1 \n 2 \n 3 \n 4 \n 0 \n 1 ]", &tflag);
      ReadCuMatrixFromString("[ 0 \n 0 \n 0 \n 0 \n 0 \n 0 \n 0 \n 0 \n 1 \n 1 \n 1 \n 1 \n 1 \n 2 \n 2 ]", &tflag);
      Vector<BaseFloat> flag;
      flag.Resize(mat_in.NumRows(), kSetZero);
      flag.CopyRowsFromMat(tflag);
      KALDI_LOG << flag.Dim();
      KALDI_LOG << flag;
      c->SetFlags(flag);
 
      CuMatrix<BaseFloat> filter;
      ReadCuMatrixFromString("[    0.1  0.2  0.3  0.4  0.5 \n\
              0.6  0.7  0.8  0.9  1.  \n\
              1.1  1.2  1.3  1.4  1.5 \n\
              1.6  1.7  1.8  1.9  2.  \n\
              2.1  2.2  2.3  2.4  2.5 \n\
              3.          3.10714286  3.21428571  3.32142857  3.42857143 \n\
              3.53571429  3.64285714  3.75        3.85714286  3.96428571 \n\
              4.07142857  4.17857143  4.28571429  4.39285714  4.5       ]", &filter);
      KALDI_LOG << filter;

      Vector<BaseFloat> para;
      int32 N1 = 4, N2 = 3, D = 5;
      para.Resize(((N1+1)+N2)*D, kSetZero);
      para.CopyRowsFromMat(filter);
      c->SetParams(para);

      // propagate,
      CuMatrix<BaseFloat> mat_out;
      c->Propagate(mat_in, &mat_out);
      KALDI_LOG << "mat_out" << mat_out;
      /* mat_out should be
         [[ 123.13571429  138.9         155.50714286  172.95714286  191.25      ]
         [ 182.27142857  200.94285714  220.65714286  241.41428571  263.21428571]
         [ 244.90714286  267.48571429  291.30714286  316.37142857  342.67857143]
         [ 313.54285714  341.02857143  369.95714286  400.32857143  432.14285714]
         [ 390.67857143  424.07142857  459.10714286  495.78571429  534.10714286]
         [ 309.28571429  338.21428571  368.57142857  400.35714286  433.57142857]
         [ 229.5         253.96428571  279.64285714  306.53571429  334.64285714]
         [ 154.          174.          195.          217.          240.        ]
         [ 591.42142857  624.04285714  657.50714286  691.81428571  726.96428571]
         [ 674.55714286  714.08571429  754.65714286  796.27142857  838.92857143]
         [ 512.47857143  548.66428571  585.87857143  624.12142857  663.39285714]
         [ 391.4         425.24285714  460.1         495.97142857  532.85714286]
         [ 316.5         349.          382.5         417.          452.5       ]
         [ 285.6         304.11428571  323.04285714  342.38571429  362.14285714]
         [ 117.7         133.3         149.3         165.7         182.5       ]]
       */

      CuMatrix<BaseFloat> mat_out_diff(mat_out);
      ReadCuMatrixFromString("[ 75.  74.  73.  72.  71. \n\
              70.  69.  68.  67.  66. \n\
              65.  64.  63.  62.  61. \n\
              60.  59.  58.  57.  56. \n\
              55.  54.  53.  52.  51. \n\
              50.  49.  48.  47.  46. \n\
              45.  44.  43.  42.  41. \n\
              40.  39.  38.  37.  36. \n\
              35.  34.  33.  32.  31. \n\
              30.  29.  28.  27.  26. \n\
              25.  24.  23.  22.  21. \n\
              20.  19.  18.  17.  16. \n\
              15.  14.  13.  12.  11. \n\
              10.   9.   8.   7.   6. \n\
              5.   4.   3.   2.   1. ]", &mat_out_diff);
      // backpropagate,
      CuMatrix<BaseFloat> mat_in_diff;
      NnetTrainOptions opts;
      opts.learn_rate = -1.0;
      c->SetTrainOptions(opts);
      c->Backpropagate(mat_in, mat_out, mat_out_diff, &mat_in_diff);
      KALDI_LOG << "mat_in_diff " << mat_in_diff;
      /* mat_in_diff should be
         [[  407.5          433.           457.5          481.           503.5       ]
         [  600.           627.92857143   654.64285714   680.14285714
         704.42857143]
         [  817.67857143   846.96428571   874.82142857   901.25         926.25      ]
         [ 1057.85714286  1087.42857143  1115.35714286  1141.64285714
         1166.28571429]
         [  898.82142857   922.98571429   945.70714286   966.98571429
         986.82142857]
         [  767.78571429   786.54285714   804.05714286   820.32857143
         835.35714286]
         [  662.25         675.6          687.90714286   699.17142857
         709.39285714]
         [  579.71428571   587.65714286   594.75714286   601.01428571
         606.42857143]
         [  147.5          153.           157.5          161.           163.5       ]
         [  199.           203.84285714   207.67142857   210.48571429
         212.28571429]
         [  269.75         272.86428571   274.95         276.00714286
         276.03571429]
         [  354.57142857   354.88571429   354.15714286   352.38571429
         349.57142857]
         [  287.03571429   284.44285714   281.00714286   276.72857143
         271.60714286]
         [   14.            13.6           12.8           11.6           10.        ]
         [   35.5           32.76428571    29.61428571    26.05          22.07142857]]
       */

      // update
      // c->Update(mat_in, mat_out_diff);
      Vector<BaseFloat> fil;
      c->GetParams(&fil);
      KALDI_LOG << fil;
      // KALDI_LOG << c->GetBackfilter();
      /* output should be
         14600.1 14645.2 14660.3 14645.4 14600.5 
         10030.6 10126.7 10198.8 10246.9 10271 
         6526.1 6673.2 6802.3 6913.4 7006.5 
         4011.6 4147.7 4269.8 4377.9 4472 
         2107.1 2232.2 2347.3 2452.4 2547.5
       */
      // KALDI_LOG << c->GetAheadfilter();
      /* output should be
         14593 14689.1 14761.2 14809.3 14833.4 
         13368.5 13515.6 13644.8 13755.9 13849 
         11994.1 12130.2 12252.3 12360.4 12454.5
       */
      delete c;
  }

  void UnitTestBiCompactVfsmn() { /* Implemented by Kaituo XU */
    Component* cp = Component::Init(
      "<BiCompactVfsmn> <InputDim> 5 <OutputDim> 5 <BackOrder> 4 <AheadOrder> 3"
    );
    BiCompactVfsmn* c = dynamic_cast<BiCompactVfsmn*>(cp);
    Component::ComponentType t = c->GetType();
    KALDI_LOG << c->TypeToMarker(t);
    KALDI_LOG << c->Info();

    CuMatrix<BaseFloat> mat_in;
    ReadCuMatrixFromString("[ 1.   2.   3.   4.   5. \n\
          6.   7.   8.   9.  10. \n\
         11.  12.  13.  14.  15. \n\
         16.  17.  18.  19.  20. \n\
         21.  22.  23.  24.  25. \n\
         26.  27.  28.  29.  30. \n\
         31.  32.  33.  34.  35. \n\
         36.  37.  38.  39.  40. \n\
         41.  42.  43.  44.  45. \n\
         46.  47.  48.  49.  50. \n\
         51.  52.  53.  54.  55. \n\
         56.  57.  58.  59.  60. \n\
         61.  62.  63.  64.  65. \n\
         66.  67.  68.  69.  70. \n\
         71.  72.  73.  74.  75. ]", &mat_in);
    KALDI_LOG << mat_in.NumRows() << " " << mat_in.NumCols();
    KALDI_LOG << mat_in;

    CuMatrix<BaseFloat> bposition, fposition;
    ReadCuMatrixFromString("[ 0 \n 1 \n 2 \n 3 \n 4 \n 5 \n 6 \n 7 \n 0 \n 1 \n 2 \n 3 \n 4 \n 0 \n 1 ]", &bposition);
    ReadCuMatrixFromString("[ 7 \n 6 \n 5 \n 4 \n 3 \n 2 \n 1 \n 0 \n 4 \n 3 \n 2 \n 1 \n 0 \n 1 \n 0 ]", &fposition);
    KALDI_LOG << bposition.NumRows() << " " << bposition.NumCols();
    KALDI_LOG << fposition.NumRows() << " " << fposition.NumCols();
    KALDI_LOG << bposition;
    KALDI_LOG << fposition;

    // Prepare extra info
    ExtraInfo info(bposition, fposition);
    c->Prepare(info);

    CuMatrix<BaseFloat> filter;
    ReadCuMatrixFromString("[    0.1  0.2  0.3  0.4  0.5 \n\
         0.6  0.7  0.8  0.9  1.  \n\
         1.1  1.2  1.3  1.4  1.5 \n\
         1.6  1.7  1.8  1.9  2.  \n\
         2.1  2.2  2.3  2.4  2.5 \n\
         3.          3.10714286  3.21428571  3.32142857  3.42857143 \n\
         3.53571429  3.64285714  3.75        3.85714286  3.96428571 \n\
         4.07142857  4.17857143  4.28571429  4.39285714  4.5       ]", &filter);
    KALDI_LOG << filter;

    Vector<BaseFloat> para;
    int32 N1 = 4, N2 = 3, D = 5;
    para.Resize(((N1+1)+N2)*D, kSetZero);
    para.CopyRowsFromMat(filter);
    c->SetParams(para);

    // propagate,
    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in, &mat_out);
    KALDI_LOG << "mat_out" << mat_out;
    /* mat_out should be
       [[ 123.13571429  138.9         155.50714286  172.95714286  191.25      ]
       [ 182.27142857  200.94285714  220.65714286  241.41428571  263.21428571]
       [ 244.90714286  267.48571429  291.30714286  316.37142857  342.67857143]
       [ 313.54285714  341.02857143  369.95714286  400.32857143  432.14285714]
       [ 390.67857143  424.07142857  459.10714286  495.78571429  534.10714286]
       [ 309.28571429  338.21428571  368.57142857  400.35714286  433.57142857]
       [ 229.5         253.96428571  279.64285714  306.53571429  334.64285714]
       [ 154.          174.          195.          217.          240.        ]
       [ 591.42142857  624.04285714  657.50714286  691.81428571  726.96428571]
       [ 674.55714286  714.08571429  754.65714286  796.27142857  838.92857143]
       [ 512.47857143  548.66428571  585.87857143  624.12142857  663.39285714]
       [ 391.4         425.24285714  460.1         495.97142857  532.85714286]
       [ 316.5         349.          382.5         417.          452.5       ]
       [ 285.6         304.11428571  323.04285714  342.38571429  362.14285714]
       [ 117.7         133.3         149.3         165.7         182.5       ]]
     */

    CuMatrix<BaseFloat> mat_out_diff(mat_out);
    ReadCuMatrixFromString("[ 75.  74.  73.  72.  71. \n\
         70.  69.  68.  67.  66. \n\
         65.  64.  63.  62.  61. \n\
         60.  59.  58.  57.  56. \n\
         55.  54.  53.  52.  51. \n\
         50.  49.  48.  47.  46. \n\
         45.  44.  43.  42.  41. \n\
         40.  39.  38.  37.  36. \n\
         35.  34.  33.  32.  31. \n\
         30.  29.  28.  27.  26. \n\
         25.  24.  23.  22.  21. \n\
         20.  19.  18.  17.  16. \n\
         15.  14.  13.  12.  11. \n\
         10.   9.   8.   7.   6. \n\
          5.   4.   3.   2.   1. ]", &mat_out_diff);
    // backpropagate,
    CuMatrix<BaseFloat> mat_in_diff;
    c->Backpropagate(mat_in, mat_out, mat_out_diff, &mat_in_diff);
    KALDI_LOG << "mat_in_diff " << mat_in_diff;
    /* mat_in_diff should be
       [[  407.5          433.           457.5          481.           503.5       ]
       [  600.           627.92857143   654.64285714   680.14285714
       704.42857143]
       [  817.67857143   846.96428571   874.82142857   901.25         926.25      ]
       [ 1057.85714286  1087.42857143  1115.35714286  1141.64285714
       1166.28571429]
       [  898.82142857   922.98571429   945.70714286   966.98571429
       986.82142857]
       [  767.78571429   786.54285714   804.05714286   820.32857143
       835.35714286]
       [  662.25         675.6          687.90714286   699.17142857
       709.39285714]
       [  579.71428571   587.65714286   594.75714286   601.01428571
       606.42857143]
       [  147.5          153.           157.5          161.           163.5       ]
       [  199.           203.84285714   207.67142857   210.48571429
       212.28571429]
       [  269.75         272.86428571   274.95         276.00714286
       276.03571429]
       [  354.57142857   354.88571429   354.15714286   352.38571429
       349.57142857]
       [  287.03571429   284.44285714   281.00714286   276.72857143
       271.60714286]
       [   14.            13.6           12.8           11.6           10.        ]
       [   35.5           32.76428571    29.61428571    26.05          22.07142857]]
    */
    
    // update
    NnetTrainOptions opts;
    opts.learn_rate = -1.0;
    c->SetTrainOptions(opts);
    c->Update(mat_in, mat_out_diff);
    KALDI_LOG << c->GetBackfilter();
    /* output should be
       14600.1 14645.2 14660.3 14645.4 14600.5 
       10030.6 10126.7 10198.8 10246.9 10271 
       6526.1 6673.2 6802.3 6913.4 7006.5 
       4011.6 4147.7 4269.8 4377.9 4472 
       2107.1 2232.2 2347.3 2452.4 2547.5
    */
    KALDI_LOG << c->GetAheadfilter();
    /* output should be
       14593 14689.1 14761.2 14809.3 14833.4 
       13368.5 13515.6 13644.8 13755.9 13849 
       11994.1 12130.2 12252.3 12360.4 12454.5
    */
    delete c;
  }

} // namespace aslp_nnet
} // namespace kaldi

int main() {
  using namespace kaldi;
  using namespace kaldi::aslp_nnet;

  for (kaldi::int32 loop = 0; loop < 2; loop++) {
#if HAVE_CUDA == 1
    if (loop == 0)
      CuDevice::Instantiate().SelectGpuId("no"); // use no GPU
    else
      CuDevice::Instantiate().SelectGpuId("optional"); // use GPU when available
#endif
    // unit-tests :
    // UnitTestLengthNorm();
    // UnitTestConvolutionalComponentUnity();
    // UnitTestConvolutionalComponent3x3();
    // UnitTestMaxPoolingComponent();
    // UnitTestFsmn();
    UnitTestBiCompactVfsmn();
    // end of unit-tests,
    if (loop == 0)
        KALDI_LOG << "Tests without GPU use succeeded.";
      else
        KALDI_LOG << "Tests with GPU use (if available) succeeded.";
  }
#if HAVE_CUDA == 1
  CuDevice::Instantiate().PrintProfile();
#endif
  return 0; 
}
