// aslp-nnet/nnet-nnet.cc

// Copyright 2011-2013  Brno University of Technology (Author: Karel Vesely)
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

#include "aslp-nnet/nnet-nnet.h"
#include "aslp-nnet/nnet-component.h"
#include "aslp-nnet/nnet-io.h"
#include "aslp-nnet/nnet-activation.h"
#include "aslp-nnet/nnet-affine-transform.h"
#include "aslp-nnet/nnet-various.h"
#include "aslp-nnet/nnet-lstm-projected-streams.h"
#include "aslp-nnet/nnet-blstm-projected-streams.h"
#include "aslp-nnet/nnet-blstm-projected-streams-lc.h"
#include "aslp-nnet/nnet-recurrent-component.h"
#include "aslp-nnet/nnet-row-convolution.h"
#include "aslp-nnet/nnet-gru-streams.h"
#include "aslp-nnet/nnet-lstm-couple-if-projected-streams.h"
#include "aslp-nnet/nnet-batch-normalization.h"
#include "aslp-nnet/nnet-cfsmn-component.h"

namespace kaldi {
namespace aslp_nnet {


Nnet::Nnet(const Nnet& other) {
  // copy the components
  for(int32 i = 0; i < other.NumComponents(); i++) {
    components_.push_back(other.GetComponent(i).Copy());
  }
  // copy train opts
  SetTrainOptions(other.opts_);
  InitInputOutput();
  Check();
}

Nnet & Nnet::operator = (const Nnet& other) {
  Destroy();
  // copy the components
  for(int32 i = 0; i < other.NumComponents(); i++) {
    components_.push_back(other.GetComponent(i).Copy());
  }
  // copy train opts
  SetTrainOptions(other.opts_); 
  InitInputOutput();
  Check();
  return *this;
}


Nnet::~Nnet() {
  Destroy();
}

void Nnet::Propagate(const std::vector<const CuMatrixBase<BaseFloat> *> &in, 
        std::vector<CuMatrix<BaseFloat> *> *out) {
    KALDI_ASSERT(NULL != out);
    
    KALDI_ASSERT(in.size() == input_.size());
    int num_frame = in[0]->NumRows();
    // 1. Resize
    for(int32 i=0; i<(int32)components_.size(); i++) {
        input_buf_[i].Resize(num_frame, components_[i]->InputDim(), kSetZero);
    }
    // 2. Copy in to InputLayer
    for (int i = 0; i < input_.size(); i++) {
        int idx = input_[i];
        input_buf_[idx].CopyFromMat(*(in[i]));
    }
    // 3. Do propagate
    for(int32 i=0; i<(int32)components_.size(); i++) {
        if (components_[i]->GetType() != Component::kInputLayer) {
            const std::vector<int32> &input_idx = components_[i]->GetInput();
            const std::vector<int32> &offset = components_[i]->GetOffset();
            KALDI_ASSERT(input_idx.size() == offset.size());
            for (int j = 0; j < input_idx.size(); j++) {
                int out_len = components_[input_idx[j]]->OutputDim();
                input_buf_[i].ColRange(offset[j], out_len).AddMat(1.0, 
                    output_buf_[input_idx[j]]);
            }
        }
        Timer tim1;
		components_[i]->Propagate(input_buf_[i], &output_buf_[i]);
		propagate_time_[i].first = Component::TypeToMarker(components_[i]->GetType());
		propagate_time_[i].second += tim1.Elapsed();
    }
    // 4. Copy to Output
    for (int i = 0; i < output_.size(); i++) {
        *((*out)[i]) = output_buf_[output_[i]];
    }
}

void Nnet::Backpropagate(const std::vector<const CuMatrixBase<BaseFloat> *> &out_diff, 
        std::vector<CuMatrix<BaseFloat> *> *in_diff) {
    KALDI_ASSERT(out_diff.size() == output_.size());
    int num_frame = out_diff[0]->NumRows();
    // 1. Resize
    for(int32 i=0; i<(int32)components_.size(); i++) {
        output_diff_buf_[i].Resize(num_frame, components_[i]->OutputDim(), kSetZero);
    }
    // 2. Copy in to output buffer
    for (int i = 0; i < output_.size(); i++) {
        int idx = output_[i];
        output_diff_buf_[idx].CopyFromMat(*(out_diff[i]));
    }
    // 3. Do BackPropagate
    for (int32 i = NumComponents()-1; i >= 0; i--) {
        Timer tim2;
		components_[i]->Backpropagate(input_buf_[i], output_buf_[i],
                            output_diff_buf_[i], &input_diff_buf_[i]);
        if (components_[i]->IsUpdatable()) {
            UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(components_[i]);
            uc->Update(input_buf_[i], output_diff_buf_[i]);
        }
		back_propagate_time_[i].first = Component::TypeToMarker(components_[i]->GetType());
		back_propagate_time_[i].second += tim2.Elapsed();

        if (components_[i]->GetType() != Component::kInputLayer) {
            const std::vector<int32> &input_idx = components_[i]->GetInput();
            const std::vector<int32> &offset = components_[i]->GetOffset();
            KALDI_ASSERT(input_idx.size() == offset.size());
            for (int j = 0; j < input_idx.size(); j++) {
                KALDI_ASSERT(input_idx[j] >= 0);
                KALDI_ASSERT(input_idx[j] <= NumComponents());
                int out_len = components_[input_idx[j]]->OutputDim();
                output_diff_buf_[input_idx[j]].AddMat(1.0, 
                        input_diff_buf_[i].ColRange(offset[j], out_len));
            }
        }
    }
    // 4. Copy to in_diff
    if (NULL == in_diff) return;
    for (int i = 0; i < input_.size(); i++) {
        if ((*in_diff)[i] != NULL) {
            *((*in_diff)[i]) = input_diff_buf_[input_[i]];
        }
    }
    //KALDI_LOG << "5. Done";
}

void Nnet::Feedforward(const std::vector<const CuMatrixBase<BaseFloat> *> &in, 
        std::vector<CuMatrix<BaseFloat> *> *out) {
    KALDI_ASSERT(NULL != out);
    
    KALDI_ASSERT(in.size() == input_.size());
    int num_frame = in[0]->NumRows();
    // 1. Resize
    for(int32 i=0; i<(int32)components_.size(); i++) {
        input_buf_[i].Resize(num_frame, components_[i]->InputDim(), kSetZero);
    }
    // 2. Copy in to InputLayer
    for (int i = 0; i < input_.size(); i++) {
        int idx = input_[i];
        input_buf_[idx].CopyFromMat(*(in[i]));
    }
    // 3. Do propagate
    for(int32 i=0; i<(int32)components_.size(); i++) {
        if (components_[i]->GetType() != Component::kInputLayer) {
            const std::vector<int32> &input_idx = components_[i]->GetInput();
            const std::vector<int32> &offset = components_[i]->GetOffset();
            KALDI_ASSERT(input_idx.size() == offset.size());
            for (int j = 0; j < input_idx.size(); j++) {
                int out_len = components_[input_idx[j]]->OutputDim();
                input_buf_[i].ColRange(offset[j], out_len).AddMat(1.0, 
                    output_buf_[input_idx[j]]);
            }
        }
        components_[i]->Feedforward(input_buf_[i], &output_buf_[i]);
    }
    // 4. Copy to Output
    for (int i = 0; i < output_.size(); i++) {
        *((*out)[i]) = output_buf_[output_[i]];
    }
}

void Nnet::Propagate(const CuMatrixBase<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    KALDI_ASSERT(NULL != out);
    if (NumComponents() == 0) {
        (*out) = in; // copy 
        return; 
    }
    KALDI_ASSERT(input_.size() == 1);
    KALDI_ASSERT(output_.size() == 1);
    std::vector<const CuMatrixBase<BaseFloat> *> in_vec;
    in_vec.push_back(&in);
    std::vector<CuMatrix<BaseFloat> *> out_vec;
    out_vec.push_back(out);
    Propagate(in_vec, &out_vec);
}

void Nnet::Backpropagate(const CuMatrixBase<BaseFloat> &out_diff, CuMatrix<BaseFloat> *in_diff) {
    // 0 layers
    if (NumComponents() == 0) { (*in_diff) = out_diff; return; }
    KALDI_ASSERT(input_.size() == 1);
    KALDI_ASSERT(output_.size() == 1);
    std::vector<const CuMatrixBase<BaseFloat> *> out_diff_vec;
    out_diff_vec.push_back(&out_diff);
    std::vector<CuMatrix<BaseFloat> *> in_diff_vec;
    in_diff_vec.push_back(in_diff);
    Backpropagate(out_diff_vec, &in_diff_vec);
}

void Nnet::Feedforward(const CuMatrixBase<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    KALDI_ASSERT(NULL != out);

    if (NumComponents() == 0) { 
        out->Resize(in.NumRows(), in.NumCols());
        out->CopyFromMat(in); 
        return; 
    }
    KALDI_ASSERT(input_.size() == 1);
    KALDI_ASSERT(output_.size() == 1);
    std::vector<const CuMatrixBase<BaseFloat> *> in_vec;
    in_vec.push_back(&in);
    std::vector<CuMatrix<BaseFloat> *> out_vec;
    out_vec.push_back(out);
    Feedforward(in_vec, &out_vec);
}

int32 Nnet::OutputDim() const {
  KALDI_ASSERT(!components_.empty());
  return components_.back()->OutputDim();
}

int32 Nnet::InputDim() const {
  KALDI_ASSERT(!components_.empty());
  return components_.front()->InputDim();
}

const Component& Nnet::GetComponent(int32 component) const {
  KALDI_ASSERT(static_cast<size_t>(component) < components_.size());
  return *(components_[component]);
}

Component& Nnet::GetComponent(int32 component) {
  KALDI_ASSERT(static_cast<size_t>(component) < components_.size());
  return *(components_[component]);
}

void Nnet::SetComponent(int32 c, Component *component) {
  KALDI_ASSERT(static_cast<size_t>(c) < components_.size());
  delete components_[c];
  components_[c] = component;
  InitInputOutput();
  Check(); // Check that all the dimensions still match up.
}

void Nnet::AppendComponent(Component* dynamically_allocated_comp) {
  // append,
  components_.push_back(dynamically_allocated_comp);
  for (int32 i = 0; i < components_.size(); i++) {
    components_[i]->SetId(i);
    components_[i]->SetMonoInput(i-1);
  }
  //
  InitInputOutput();
  //Check();
}

void Nnet::AppendNnet(const Nnet& nnet_to_append) {
  // append,
  for(int32 i=0; i<nnet_to_append.NumComponents(); i++) {
    AppendComponent(nnet_to_append.GetComponent(i).Copy());
  }
  //
  InitInputOutput();
  Check();
}

void Nnet::RemoveComponent(int32 component) {
  KALDI_ASSERT(component < NumComponents());
  // remove,
  Component* ptr = components_[component];
  components_.erase(components_.begin()+component);
  delete ptr;
  // 
  InitInputOutput();
  Check();
}


void Nnet::GetParams(Vector<BaseFloat>* wei_copy) const {
  wei_copy->Resize(NumParams());
  int32 pos = 0;
  // copy the params
  for(int32 i=0; i<components_.size(); i++) {
    if(components_[i]->IsUpdatable()) {
      UpdatableComponent& c = dynamic_cast<UpdatableComponent&>(*components_[i]);
      Vector<BaseFloat> c_params; 
      c.GetParams(&c_params);
      wei_copy->Range(pos,c_params.Dim()).CopyFromVec(c_params);
      pos += c_params.Dim();
    }
  }
  KALDI_ASSERT(pos == NumParams());
}


void Nnet::GetGpuParams(std::vector<std::pair<BaseFloat *, int> > *params) {
  KALDI_ASSERT(params != NULL);
  params->clear();
  for(int32 i=0; i<components_.size(); i++) {
    if(components_[i]->IsUpdatable()) {
      UpdatableComponent& c = dynamic_cast<UpdatableComponent&>(*components_[i]);
      std::vector<std::pair<BaseFloat *, int> > c_params;
      c.GetGpuParams(&c_params);
      params->insert(params->end(), c_params.begin(), c_params.end());
    }
  }
}

void Nnet::GetAccStats(std::vector<double *> *acc_params, 
                       std::vector<std::pair<double*, int> > *data_params) {
  KALDI_ASSERT(acc_params != NULL);
  KALDI_ASSERT(data_params != NULL);
  acc_params->clear();
  data_params->clear();
  for(int32 i=0; i<components_.size(); i++) {
    if (GetComponent(i).GetType() == Component::kBatchNormalization) {
      BatchNormalization& bat_norm = dynamic_cast<BatchNormalization&>(GetComponent(i));
      std::vector<std::pair<double*, int> > c_params;
      double *acc_ptr = bat_norm.GetAccStats(&c_params);
      acc_params->push_back(acc_ptr);
      data_params->insert(data_params->end(), c_params.begin(), c_params.end());
    }
  }
}

void Nnet::GetWeights(Vector<BaseFloat>* wei_copy) const {
  wei_copy->Resize(NumParams());
  int32 pos = 0;
  // copy the params
  for(int32 n=0; n<components_.size(); n++) {
    if(components_[n]->IsUpdatable()) {
      switch(components_[n]->GetType()) {
        case Component::kAffineTransform : {
          // copy weight matrix row-by-row to the vector
          Matrix<BaseFloat> mat(dynamic_cast<AffineTransform*>(components_[n])->GetLinearity());
          int32 mat_size = mat.NumRows()*mat.NumCols();
          wei_copy->Range(pos,mat_size).CopyRowsFromMat(mat);
          pos += mat_size;
          // append biases
          Vector<BaseFloat> vec(dynamic_cast<AffineTransform*>(components_[n])->GetBias());
          wei_copy->Range(pos,vec.Dim()).CopyFromVec(vec);
          pos += vec.Dim();
        } break;
        default :
          KALDI_ERR << "Unimplemented access to parameters "
                    << "of updatable component " 
                    << Component::TypeToMarker(components_[n]->GetType());
      }
    }
  }
  KALDI_ASSERT(pos == NumParams());
}


void Nnet::SetWeights(const Vector<BaseFloat>& wei_src) {
  KALDI_ASSERT(wei_src.Dim() == NumParams());
  int32 pos = 0;
  for(int32 n=0; n<components_.size(); n++) {
    if(components_[n]->IsUpdatable()) {
      switch(components_[n]->GetType()) {
        case Component::kAffineTransform : {
          // get the component
          AffineTransform* aff_t = dynamic_cast<AffineTransform*>(components_[n]);
          // we need weight matrix with original dimensions
          Matrix<BaseFloat> mat(aff_t->GetLinearity());
          int32 mat_size = mat.NumRows()*mat.NumCols();
          mat.CopyRowsFromVec(wei_src.Range(pos,mat_size));
          pos += mat_size;
          // get the bias vector
          Vector<BaseFloat> vec(aff_t->GetBias());
          vec.CopyFromVec(wei_src.Range(pos,vec.Dim()));
          pos += vec.Dim();
          // assign to the component
          aff_t->SetLinearity(CuMatrix<BaseFloat>(mat));
          aff_t->SetBias(CuVector<BaseFloat>(vec));
        } break;
        default :
          KALDI_ERR << "Unimplemented access to parameters "
                    << "of updatable component " 
                    << Component::TypeToMarker(components_[n]->GetType());
      }
    }
  }
  KALDI_ASSERT(pos == NumParams());
}

 
void Nnet::GetGradient(Vector<BaseFloat>* grad_copy) const {
  grad_copy->Resize(NumParams());
  int32 pos = 0;
  // copy the params
  for(int32 n=0; n<components_.size(); n++) {
    if(components_[n]->IsUpdatable()) {
      switch(components_[n]->GetType()) {
        case Component::kAffineTransform : {
          // get the weights from CuMatrix to Matrix
          const CuMatrixBase<BaseFloat>& cu_mat = 
            dynamic_cast<AffineTransform*>(components_[n])->GetLinearityCorr();
          Matrix<BaseFloat> mat(cu_mat.NumRows(),cu_mat.NumCols());
          cu_mat.CopyToMat(&mat);
          // copy the the matrix row-by-row to the vector
          int32 mat_size = mat.NumRows()*mat.NumCols();
          grad_copy->Range(pos,mat_size).CopyRowsFromMat(mat);
          pos += mat_size;
          // get the biases from CuVector to Vector
          const CuVector<BaseFloat>& cu_vec = 
            dynamic_cast<AffineTransform*>(components_[n])->GetBiasCorr();
          Vector<BaseFloat> vec(cu_vec.Dim());
          cu_vec.CopyToVec(&vec);
          // append biases to the supervector
          grad_copy->Range(pos,vec.Dim()).CopyFromVec(vec);
          pos += vec.Dim();
        } break;
        default :
          KALDI_ERR << "Unimplemented access to parameters "
                    << "of updatable component " 
                    << Component::TypeToMarker(components_[n]->GetType());
      }
    }
  }
  KALDI_ASSERT(pos == NumParams());
}


int32 Nnet::NumParams() const {
  int32 n_params = 0;
  for(int32 n=0; n<components_.size(); n++) {
    if(components_[n]->IsUpdatable()) {
      n_params += dynamic_cast<UpdatableComponent*>(components_[n])->NumParams();
    }
  }
  return n_params;
}


void Nnet::SetDropoutRetention(BaseFloat r)  {
  for (int32 c=0; c < NumComponents(); c++) {
    if (GetComponent(c).GetType() == Component::kDropout) {
      Dropout& comp = dynamic_cast<Dropout&>(GetComponent(c));
      BaseFloat r_old = comp.GetDropoutRetention();
      comp.SetDropoutRetention(r);
      KALDI_LOG << "Setting dropout-retention in component " << c 
                << " from " << r_old << " to " << r;
    }
  }
}


void Nnet::ResetLstmStreams(const std::vector<int32> &stream_reset_flag) {
  for (int32 c=0; c < NumComponents(); c++) {
    if (GetComponent(c).GetType() == Component::kLstmProjectedStreams) {
      LstmProjectedStreams& comp = dynamic_cast<LstmProjectedStreams&>(GetComponent(c));
      comp.ResetLstmStreams(stream_reset_flag);
    }    
    else if (GetComponent(c).GetType() == Component::kLstm) {
      Lstm& comp = dynamic_cast<Lstm&>(GetComponent(c));
      comp.ResetLstmStreams(stream_reset_flag);
    }
    else if (GetComponent(c).GetType() == Component::kBLstmProjectedStreamsLC) {
      BLstmProjectedStreamsLC& comp = dynamic_cast<BLstmProjectedStreamsLC&>(GetComponent(c));
      comp.ResetLstmStreams(stream_reset_flag);
    }
    else if (GetComponent(c).GetType() == Component::kGruStreams) {
      GruStreams& comp = dynamic_cast<GruStreams&>(GetComponent(c));
      comp.ResetLstmStreams(stream_reset_flag);
    }
    else if (GetComponent(c).GetType() == Component::kLstmCifgProjectedStreams) {
      LstmCifgProjectedStreams& comp = dynamic_cast<LstmCifgProjectedStreams&>(GetComponent(c));
      comp.ResetLstmStreams(stream_reset_flag);
    }
  }
}

void Nnet::SetSeqLengths(const std::vector<int32> &sequence_lengths) {
  for (int32 c=0; c < NumComponents(); c++) {
    if (GetComponent(c).GetType() == Component::kBLstmProjectedStreams) {
      BLstmProjectedStreams& comp = dynamic_cast<BLstmProjectedStreams&>(GetComponent(c));
      comp.SetSeqLengths(sequence_lengths);
    }
    else if (GetComponent(c).GetType() == Component::kBLstm) {
      BLstm &comp = dynamic_cast<BLstm&>(GetComponent(c));
      comp.SetSeqLengths(sequence_lengths);
    }
    else if (GetComponent(c).GetType() == Component::kLstmProjectedStreams) {
      LstmProjectedStreams &comp = dynamic_cast<LstmProjectedStreams&>(GetComponent(c));
      comp.SetSeqLengths(sequence_lengths);
    }
    else if (GetComponent(c).GetType() == Component::kLstm) {
      Lstm &comp = dynamic_cast<Lstm&>(GetComponent(c));
      comp.SetSeqLengths(sequence_lengths);
    }
    else if (GetComponent(c).GetType() == Component::kRowConvolution) {
      RowConvolution &comp = dynamic_cast<RowConvolution&>(GetComponent(c));
      comp.SetSeqLengths(sequence_lengths);
    }
    else if (GetComponent(c).GetType() == Component::kGruStreams) {
      GruStreams &comp = dynamic_cast<GruStreams&>(GetComponent(c));
      comp.SetSeqLengths(sequence_lengths);
    }
    else if (GetComponent(c).GetType() == Component::kLstmCifgProjectedStreams) {
      LstmCifgProjectedStreams &comp = dynamic_cast<LstmCifgProjectedStreams&>(GetComponent(c));
      comp.SetSeqLengths(sequence_lengths);
    }
  }
}

void Nnet::SetChunkSize(int chunk_size) {
  for (int32 c=0; c < NumComponents(); c++) {
    if (GetComponent(c).GetType() == Component::kBLstmProjectedStreamsLC) {
      BLstmProjectedStreamsLC& comp = dynamic_cast<BLstmProjectedStreamsLC&>(GetComponent(c));
      comp.SetChunkSize(chunk_size);
    }
  }
}

void Nnet::AutoComplete() {
    // Optional add InputLayer
    int input_dim = components_[0]->InputDim();
    Component *in_comp = new InputLayer(input_dim, input_dim);
    in_comp->SetId(0);
    in_comp->SetMonoInput(-1);
    components_.insert(components_.begin(), in_comp);
    KALDI_ASSERT(components_[0]->Id() == 0);
    
    // Auto assign id and input
    for (int i = 1; i < components_.size(); i++) {
        //KALDI_LOG << components_[i]->Id();
        KALDI_ASSERT(components_[i]->Id() < 0);
        components_[i]->SetId(i);
        components_[i]->SetMonoInput(i-1);
    }

    // Optional add OutputLayer
    int num_layers = components_.size();
    int output_dim = components_[num_layers - 1]->OutputDim();
    Component *out_comp = new OutputLayer(output_dim, output_dim);
    out_comp->SetId(num_layers);
    out_comp->SetMonoInput(num_layers-1);
    components_.push_back(out_comp);
    KALDI_ASSERT(components_[components_.size()-1]->Id() == components_.size()-1);
}

void Nnet::Init(const std::string &file) {
  Input in(file);
  std::istream &is = in.Stream();
  // do the initialization with config lines,
  std::string conf_line, token;
  bool simple_net = true;
  while (!is.eof()) {
    KALDI_ASSERT(is.good());
    std::getline(is, conf_line); // get a line from config file,
    if (conf_line == "") continue;
    KALDI_VLOG(1) << conf_line; 
    std::istringstream is(conf_line);
	is >> std::ws >> token; // get 1st token,
	if (token == "<NnetProto>" || token == "</NnetProto>") continue; // ignored tokens,
    if (token == "<StructureType>") {
		is >> std::ws >> token;
		if (token == "graph") {
			simple_net = false;
			//std::vector<std::vector<std::pair<std::string, int32> > > nnet_map;
		}
		else if (token == "simple") simple_net = true;
		else KALDI_ERR << "The net's structure must be simple or graphi!";
		continue;
	}
	
	//AppendComponent(Component::Init(conf_line+"\n"));
    Component *comp = Component::Init(conf_line+"\n");
    components_.push_back(comp);
	
    is >> std::ws;
  }
  if (!simple_net) {
	AssignComponentId(components_);
	SortComponent(components_);
  }

  // Automatic assgin id and input for simple net
  if (simple_net) AutoComplete();
  // cleanup
  in.Close();
  InitInputOutput();
  Check();
}


void Nnet::Read(const std::string &file) {
  bool binary;
  Input in(file, &binary);
  Read(in.Stream(), binary);
  in.Close();
  // Warn if the NN is empty
  if(NumComponents() == 0) {
    KALDI_WARN << "The network '" << file << "' is empty.";
  }
}


void Nnet::Read(std::istream &is, bool binary) {
  // get the network layers from a factory
  Component *comp;
  while (NULL != (comp = Component::Read(is, binary))) {
    int id = comp->Id();
    if (id >= components_.size()) components_.resize(id+1, NULL);
    if (components_[id] != NULL) {
      KALDI_ERR << "Component id " << id << " already be taken" 
                << "the id must be unique"; 
    }
    // components i index = comp->Id(), in order
    components_[id] = comp;
  }
  // reset learn rate
  opts_.learn_rate = 0.0;
  
  InitInputOutput();
  Check(); //check consistency (dims...)
}

void Nnet::Write(const std::string &file, bool binary) const {
  Output out(file, binary, true);
  Write(out.Stream(), binary);
  out.Close();
}

void Nnet::Write(std::ostream &os, bool binary) const {
  Check();
  WriteToken(os, binary, "<Nnet>");
  if(binary == false) os << std::endl;
  for(int32 i=0; i<NumComponents(); i++) {
    components_[i]->Write(os, binary);
  }
  WriteToken(os, binary, "</Nnet>");  
  if(binary == false) os << std::endl;
}

void Nnet::WriteDotFile(std::ofstream &os) const {
	if (!os.is_open())
		KALDI_ERR << "Open dot file fialed!";
	os << "digraph net{" << std::endl;
	os << "rankdir=BT" << std::endl;
	os << "node[shape = box; height = 1; width = 3; fontsize = 40];" << std::endl;
	os << "edge[minlen = 1 ]" << std::endl;
	for (int32 i = 0; i < NumComponents(); i++) {
		std::string name = components_[i]->GetName();
		int id = components_[i]->Id();
		std::vector<int32> input = components_[i]->GetInput();
		std::vector<int32> offset = components_[i]->GetOffset();
		if (name != "")
			os << id << " [label = " << "\"" << name << "\"]" << std::endl;
		else
			os << id << " [label = " << "\"" << Component::TypeToMarker(components_[i]->GetType()) << "\"]" << std::endl;
		if (input.size() == 1 && input[0] == -1) continue;
		for (int32 j = 0; j < input.size(); j++) {
			os << "\t" << input[j] << " -> " << id << " [label = " << offset[j] << "; fontsize = 40]" << std::endl;
		}

	}

	os << "}" << std::endl;
}

void Nnet::WriteStandard(const std::string &file, bool binary) const {
  Output out(file, binary, true);
  Write(out.Stream(), binary);
  out.Close();
}

void Nnet::WriteStandard(std::ostream &os, bool binary) const {
  Check();
  WriteToken(os, binary, "<Nnet>");
  if(binary == false) os << std::endl;
  for(int32 i=0; i<NumComponents(); i++) {
    if (components_[i]->GetType() == Component::kInputLayer || 
        components_[i]->GetType() == Component::kOutputLayer) continue;
    components_[i]->WriteStandard(os, binary);
  }
  WriteToken(os, binary, "</Nnet>");  
  if(binary == false) os << std::endl;
}

std::string Nnet::Info() const {
  // global info
  std::ostringstream ostr;
  ostr << "num-components " << NumComponents() << std::endl;
  ostr << "input-dim " << InputDim() << std::endl;
  ostr << "output-dim " << OutputDim() << std::endl;
  ostr << "number-of-parameters " << static_cast<float>(NumParams())/1e6 
       << " millions" << std::endl;
  // topology & weight stats
  for (int32 i = 0; i < NumComponents(); i++) {
    ostr << "component " << i+1 << " : " 
         << Component::TypeToMarker(components_[i]->GetType()) 
         << ", input-dim " << components_[i]->InputDim()
         << ", output-dim " << components_[i]->OutputDim()
         << ", id " << components_[i]->Id();
    const std::vector<int32> &input_idx = components_[i]->GetInput();
    const std::vector<int32> &offset = components_[i]->GetOffset();
    ostr << ", input ";
    for (int j = 0; j < input_idx.size(); j++) {
      ostr << input_idx[j] << ":" << offset[j] << ",";
    }
    ostr << "  " << components_[i]->Info() << std::endl;
  }
  return ostr.str();
}

std::string Nnet::InfoGradient() const {
  std::ostringstream ostr;
  // gradient stats
  ostr << "### Gradient stats :\n";
  for (int32 i = 0; i < NumComponents(); i++) {
    ostr << "Component " << i+1 << " : " 
         << Component::TypeToMarker(components_[i]->GetType()) 
         << ", " << components_[i]->InfoGradient() << std::endl;
  }
  return ostr.str();
}

std::string Nnet::InfoPropagate() const {
  std::ostringstream ostr;
  // forward-pass buffer stats
  ostr << "### Forward propagation buffer content :\n";
  ostr << "[0] output of <Input> " << MomentStatistics(input_buf_[0]) << std::endl;
  for (int32 i=0; i<NumComponents(); i++) {
    ostr << "["<<1+i<< "] output of " 
         << Component::TypeToMarker(components_[i]->GetType())
         << MomentStatistics(output_buf_[i]) << std::endl;
    // nested networks too...
    //if (Component::kParallelComponent == components_[i]->GetType()) {
    //  ostr << dynamic_cast<ParallelComponent*>(components_[i])->InfoPropagate();
    //}
  }
  return ostr.str();
}

std::string Nnet::InfoBackPropagate() const {
  std::ostringstream ostr;
  // forward-pass buffer stats
  ostr << "### Backward propagation buffer content :\n";
  ostr << "[0] diff of <Input> " << MomentStatistics(output_diff_buf_[0]) << std::endl;
  for (int32 i=0; i<NumComponents(); i++) {
    ostr << "["<<1+i<< "] diff-output of " 
         << Component::TypeToMarker(components_[i]->GetType())
         << MomentStatistics(output_diff_buf_[i]) << std::endl;
    // nested networks too...
    //if (Component::kParallelComponent == components_[i]->GetType()) {
    //  ostr << dynamic_cast<ParallelComponent*>(components_[i])->InfoBackPropagate();
    //}
  }
  return ostr.str();
}

void Nnet::Check() const {
  // Check must have input and output
  if (input_.size() < 1) {
    KALDI_ERR << "Must have at least one InputLayer";
  }
  if (output_.size() < 1) {
    KALDI_ERR << "Must have at least one OutputLayer";
  }
  // Check index == id && none null component
  for (int i = 0; i < NumComponents(); i++) {
    if (components_[i] == NULL) {
      KALDI_ERR << "Component id must be consistant, but have no id " << i;
    }
    if (components_[i]->Id() != i) {
      KALDI_ERR << "Component id not equal index id, May be error in Read";
    }
  }
  // Check layer's input id must less than layer' id
  for (int i = 0; i < NumComponents(); i++) {
    if (components_[i]->GetType() == Component::kInputLayer) continue;
    const std::vector<int32> &input_idx = components_[i]->GetInput();
    const std::vector<int32> &offset = components_[i]->GetOffset();
    KALDI_ASSERT(input_idx.size() == offset.size());
    for (int j = 0; j < input_idx.size(); j++) {
      int idx = input_idx[j];
      if (components_[idx]->Id() >= components_[i]->Id()) {
        KALDI_ERR << "Input id must be less than Component id, case " 
                  << " <Id> " << i << " <Input> " << idx;
      }
      int32 out_dim = components_[idx]->OutputDim();
      if (offset[j] + out_dim > components_[i]->InputDim()) {
        KALDI_ERR << "Component " << idx << " outputdim + offset must be less than "
                  << "offset " << offset[j] << " "
                  << "outdim " << out_dim << " "
                  << "Component " << i << " inputdim";
      }
    }
  }
  // check for nan/inf in network weights,
  Vector<BaseFloat> weights;
  GetParams(&weights);
  BaseFloat sum = weights.Sum();
  if(KALDI_ISINF(sum)) {
    KALDI_ERR << "'inf' in network parameters (weight explosion, try lower learning rate?)";
  }
  if(KALDI_ISNAN(sum)) {
    KALDI_ERR << "'nan' in network parameters (try lower learning rate?)";
  }
}


void Nnet::Destroy() {
  for(int32 i=0; i<NumComponents(); i++) {
    delete components_[i];
  }
  components_.resize(0);
  input_buf_.resize(0);
  input_diff_buf_.resize(0);
  output_buf_.resize(0);
  output_diff_buf_.resize(0);
}


void Nnet::SetTrainOptions(const NnetTrainOptions& opts) {
  opts_ = opts;
  //set values to individual components
  for (int32 l=0; l<NumComponents(); l++) {
    if(GetComponent(l).IsUpdatable()) {
      dynamic_cast<UpdatableComponent&>(GetComponent(l)).SetTrainOptions(opts_);
    }
  }
}

void Nnet::InitInputOutput() {
    // Reset input_ and output_
    input_.clear();
    output_.clear();
    for (int i = 0; i < NumComponents(); i++) {
        if (components_[i] == NULL) continue;
        if (components_[i]->GetType() == Component::kInputLayer) {
            input_.push_back(components_[i]->Id());
        }
        else if (components_[i]->GetType() == Component::kOutputLayer) {
            output_.push_back(components_[i]->Id());
        }
    } 
    // create empty buffers
    input_buf_.resize(NumComponents());
    output_buf_.resize(NumComponents());
    input_diff_buf_.resize(NumComponents());
    output_diff_buf_.resize(NumComponents());

	propagate_time_.resize(NumComponents());
	back_propagate_time_.resize(NumComponents());
	for (int i = 0; i < NumComponents(); i++) {
		propagate_time_[i].second = 0.0;
		back_propagate_time_[i].second = 0.0;
	}
}

void Nnet::GetComponentTime() {
	for (int i = 0; i < propagate_time_.size(); i++) {
		KALDI_LOG << propagate_time_[i].first
				  <<": Propagate time "
				  << propagate_time_[i].second << "s, "
				  << "Back-Propagate time "
				  << back_propagate_time_[i].second << "s, "
				  << "total time "
				  << propagate_time_[i].second +  back_propagate_time_[i].second << "s";
		propagate_time_[i].second = 0.0;
		back_propagate_time_[i].second = 0.0;
	}
}

void Nnet::AssignComponentId(std::vector<Component*> &comp) {
	int32 num_comp = comp.size();
	std::vector<int32> indegree;
	std::vector<std::string> input_name;
	for (int i = 0; i < num_comp; i++) {
		input_name = comp[i]->GetInputName();
		if (input_name.size() == 1 && input_name[0] == "-1") {
			indegree.push_back(0);
		}
		else {
			indegree.push_back(input_name.size());		
		}
	}
	std::vector<std::string> comp_queue;
	for (int i = 0; i < num_comp; i++) {
		if(indegree[i] == 0)
			comp_queue.push_back(comp[i]->GetName());
	}
	int32 id = 0;
	while(!comp_queue.empty()) {
		std::string name = comp_queue.back();
		comp_queue.pop_back();
		for (int i = 0; i < num_comp; i++) {
			input_name = comp[i]->GetInputName();
			if (comp[i]->GetName() == name) {
				comp[i]->SetId(id);
				++id;
			}
			for (int j = 0; j < input_name.size(); j++) {
				if (input_name[j] == comp[i]->GetName())
					KALDI_ERR << "The input of component " << comp[i]->GetName() 
						      << "include itself, Please check it!";
				else if (input_name[j] == name)
						if (--indegree[i] == 0)
							comp_queue.push_back(comp[i]->GetName());
			}
		}
	}	
	
	if (id != num_comp)
		KALDI_ERR << "The graph has a cycle";
	
	std::map<std::string, int32> name_to_id;
	for (int i = 0; i < num_comp; i++) {
		name_to_id.insert(make_pair(comp[i]->GetName(), comp[i]->GetId()));
	}
	
	for (int i = 0; i < num_comp; i++) {
		input_name = comp[i]->GetInputName();
		int32 num_input = input_name.size();
		std::vector<int32> input(num_input, 0);
		for (int j = 0; j < num_input; j++) {
			if (input_name[j] == "-1")
				input[j] = -1;
			else {
				for (std::map<std::string, int32>::iterator iter = name_to_id.begin(); iter != name_to_id.end(); ++iter) {
					if (input_name[j] == iter->first)
						input[j] = iter->second;
				}
			}
		}
		comp[i]->SetInput(input);
	}
}

void Nnet::SortComponent(std::vector<Component*> &comp) {
	std::vector<Component*> tmp_comp(comp.size());
	for (int i = 0; i < comp.size(); i++) {
		int id = comp[i]->GetId();
		tmp_comp[id] = comp[i];
	}
	comp.swap(tmp_comp);
}

void Nnet::SetFlags(const Vector<BaseFloat> &flags) {    
  for (int32 c = 0; c < NumComponents(); c++) {
    // if (GetComponent(c).GetType() == Component::kFsmn) {
    //   Fsmn& comp = dynamic_cast<Fsmn&>(GetComponent(c));
    //   comp.SetFlags(flags);
    // }
    if (GetComponent(c).GetType() == Component::kDeepFsmn) {
      DeepFsmn& comp = dynamic_cast<DeepFsmn&>(GetComponent(c));
      comp.SetFlags(flags);
    }
    // if (GetComponent(c).GetType() == Component::kUniFsmn) {
    //   UniFsmn& comp = dynamic_cast<UniFsmn&>(GetComponent(c));
    //   comp.SetFlags(flags);
    // }
    // if (GetComponent(c).GetType() == Component::kUniDeepFsmn) {
    //   UniDeepFsmn& comp = dynamic_cast<UniDeepFsmn&>(GetComponent(c));
    //   comp.SetFlags(flags);
    // }
  }
}

} // namespace aslp_nnet
} // namespace kaldi
