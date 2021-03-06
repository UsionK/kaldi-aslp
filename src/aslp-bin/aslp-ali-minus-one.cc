// aslp-bin/aslp-ali-minus-one.cc

// Copyright 2009-2011  Microsoft Corporation
// Copyright 2016 ASLP (Binbin Zhang)

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

/** @brief Converts alignments (containing transition-ids) to pdf-ids, zero-based.
*/
#include "base/kaldi-common.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "util/common-utils.h"
#include "fst/fstlib.h"

int main(int argc, char *argv[]) {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    try {
        const char *usage =
            "Minus 1 for the input alignment\n"
            "Usage:  aslp-ali-minus-one  [options] <ali-rspecifier> <ali-wspecifier>\n"
            "e.g.: \n"
            " aslp-ali-minus-one ark:1.ali ark, t:-\n";
        ParseOptions po(usage);

        bool unique = false;
        po.Register("unique", &unique, "Remove consistant duplicated alignment, only keep one");

        po.Read(argc, argv);

        if (po.NumArgs() != 2) {
            po.PrintUsage();
            exit(1);
        }

        std::string alignments_rspecifier = po.GetArg(1),
            pdfs_wspecifier = po.GetArg(2);

        SequentialInt32VectorReader reader(alignments_rspecifier);

        Int32VectorWriter writer(pdfs_wspecifier);
        int32 num_done = 0;
        for (; !reader.Done(); reader.Next()) {
            std::string key = reader.Key();
            std::vector<int32> alignment = reader.Value();
            for (size_t i = 0; i < alignment.size(); i++) {
                alignment[i] -= 1;
                KALDI_ASSERT(alignment[i] >= 0);
            }
            if (unique) {
                std::vector<int32>::iterator it = std::unique(alignment.begin(), alignment.end());
                alignment.erase(it, alignment.end());
            }
            writer.Write(key, alignment);
            num_done++;
        }
        KALDI_LOG << "Converted " << num_done << " alignments to pdf sequences.";
    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}


