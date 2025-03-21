// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: route_guide.proto

#include "route_guide.pb.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>

PROTOBUF_PRAGMA_INIT_SEG
namespace routeguide {
constexpr Jsonfile::Jsonfile(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : contents_(&::PROTOBUF_NAMESPACE_ID::internal::fixed_address_empty_string){}
struct JsonfileDefaultTypeInternal {
  constexpr JsonfileDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~JsonfileDefaultTypeInternal() {}
  union {
    Jsonfile _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT JsonfileDefaultTypeInternal _Jsonfile_default_instance_;
constexpr Results::Results(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : res_(&::PROTOBUF_NAMESPACE_ID::internal::fixed_address_empty_string){}
struct ResultsDefaultTypeInternal {
  constexpr ResultsDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~ResultsDefaultTypeInternal() {}
  union {
    Results _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT ResultsDefaultTypeInternal _Results_default_instance_;
}  // namespace routeguide
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_route_5fguide_2eproto[2];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_route_5fguide_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_route_5fguide_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_route_5fguide_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::routeguide::Jsonfile, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::routeguide::Jsonfile, contents_),
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::routeguide::Results, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::routeguide::Results, res_),
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, -1, sizeof(::routeguide::Jsonfile)},
  { 7, -1, -1, sizeof(::routeguide::Results)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::routeguide::_Jsonfile_default_instance_),
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::routeguide::_Results_default_instance_),
};

const char descriptor_table_protodef_route_5fguide_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n\021route_guide.proto\022\nrouteguide\"\034\n\010Jsonf"
  "ile\022\020\n\010contents\030\001 \001(\t\"\026\n\007Results\022\013\n\003res\030"
  "\001 \001(\t2B\n\nRouteGuide\0224\n\005infer\022\024.routeguid"
  "e.Jsonfile\032\023.routeguide.Results\"\000B6\n\033io."
  "grpc.examples.routeguideB\017RouteGuideProt"
  "oP\001\242\002\003RTGb\006proto3"
  ;
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_route_5fguide_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_route_5fguide_2eproto = {
  false, false, 217, descriptor_table_protodef_route_5fguide_2eproto, "route_guide.proto", 
  &descriptor_table_route_5fguide_2eproto_once, nullptr, 0, 2,
  schemas, file_default_instances, TableStruct_route_5fguide_2eproto::offsets,
  file_level_metadata_route_5fguide_2eproto, file_level_enum_descriptors_route_5fguide_2eproto, file_level_service_descriptors_route_5fguide_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_route_5fguide_2eproto_getter() {
  return &descriptor_table_route_5fguide_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_route_5fguide_2eproto(&descriptor_table_route_5fguide_2eproto);
namespace routeguide {

// ===================================================================

class Jsonfile::_Internal {
 public:
};

Jsonfile::Jsonfile(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor();
  if (!is_message_owned) {
    RegisterArenaDtor(arena);
  }
  // @@protoc_insertion_point(arena_constructor:routeguide.Jsonfile)
}
Jsonfile::Jsonfile(const Jsonfile& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  contents_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (!from._internal_contents().empty()) {
    contents_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_contents(), 
      GetArenaForAllocation());
  }
  // @@protoc_insertion_point(copy_constructor:routeguide.Jsonfile)
}

void Jsonfile::SharedCtor() {
contents_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}

Jsonfile::~Jsonfile() {
  // @@protoc_insertion_point(destructor:routeguide.Jsonfile)
  if (GetArenaForAllocation() != nullptr) return;
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

inline void Jsonfile::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  contents_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}

void Jsonfile::ArenaDtor(void* object) {
  Jsonfile* _this = reinterpret_cast< Jsonfile* >(object);
  (void)_this;
}
void Jsonfile::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void Jsonfile::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void Jsonfile::Clear() {
// @@protoc_insertion_point(message_clear_start:routeguide.Jsonfile)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  contents_.ClearToEmpty();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* Jsonfile::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // string contents = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10)) {
          auto str = _internal_mutable_contents();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "routeguide.Jsonfile.contents"));
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      default:
        goto handle_unusual;
    }  // switch
  handle_unusual:
    if ((tag == 0) || ((tag & 7) == 4)) {
      CHK_(ptr);
      ctx->SetLastTag(tag);
      goto message_done;
    }
    ptr = UnknownFieldParse(
        tag,
        _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
        ptr, ctx);
    CHK_(ptr != nullptr);
  }  // while
message_done:
  return ptr;
failure:
  ptr = nullptr;
  goto message_done;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* Jsonfile::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:routeguide.Jsonfile)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // string contents = 1;
  if (!this->_internal_contents().empty()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_contents().data(), static_cast<int>(this->_internal_contents().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "routeguide.Jsonfile.contents");
    target = stream->WriteStringMaybeAliased(
        1, this->_internal_contents(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:routeguide.Jsonfile)
  return target;
}

size_t Jsonfile::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:routeguide.Jsonfile)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // string contents = 1;
  if (!this->_internal_contents().empty()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_contents());
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData Jsonfile::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSizeCheck,
    Jsonfile::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*Jsonfile::GetClassData() const { return &_class_data_; }

void Jsonfile::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to,
                      const ::PROTOBUF_NAMESPACE_ID::Message& from) {
  static_cast<Jsonfile *>(to)->MergeFrom(
      static_cast<const Jsonfile &>(from));
}


void Jsonfile::MergeFrom(const Jsonfile& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:routeguide.Jsonfile)
  GOOGLE_DCHECK_NE(&from, this);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (!from._internal_contents().empty()) {
    _internal_set_contents(from._internal_contents());
  }
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void Jsonfile::CopyFrom(const Jsonfile& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:routeguide.Jsonfile)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Jsonfile::IsInitialized() const {
  return true;
}

void Jsonfile::InternalSwap(Jsonfile* other) {
  using std::swap;
  auto* lhs_arena = GetArenaForAllocation();
  auto* rhs_arena = other->GetArenaForAllocation();
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      &contents_, lhs_arena,
      &other->contents_, rhs_arena
  );
}

::PROTOBUF_NAMESPACE_ID::Metadata Jsonfile::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_route_5fguide_2eproto_getter, &descriptor_table_route_5fguide_2eproto_once,
      file_level_metadata_route_5fguide_2eproto[0]);
}

// ===================================================================

class Results::_Internal {
 public:
};

Results::Results(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor();
  if (!is_message_owned) {
    RegisterArenaDtor(arena);
  }
  // @@protoc_insertion_point(arena_constructor:routeguide.Results)
}
Results::Results(const Results& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  res_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (!from._internal_res().empty()) {
    res_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_res(), 
      GetArenaForAllocation());
  }
  // @@protoc_insertion_point(copy_constructor:routeguide.Results)
}

void Results::SharedCtor() {
res_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}

Results::~Results() {
  // @@protoc_insertion_point(destructor:routeguide.Results)
  if (GetArenaForAllocation() != nullptr) return;
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

inline void Results::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  res_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}

void Results::ArenaDtor(void* object) {
  Results* _this = reinterpret_cast< Results* >(object);
  (void)_this;
}
void Results::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void Results::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void Results::Clear() {
// @@protoc_insertion_point(message_clear_start:routeguide.Results)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  res_.ClearToEmpty();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* Results::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // string res = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10)) {
          auto str = _internal_mutable_res();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "routeguide.Results.res"));
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      default:
        goto handle_unusual;
    }  // switch
  handle_unusual:
    if ((tag == 0) || ((tag & 7) == 4)) {
      CHK_(ptr);
      ctx->SetLastTag(tag);
      goto message_done;
    }
    ptr = UnknownFieldParse(
        tag,
        _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
        ptr, ctx);
    CHK_(ptr != nullptr);
  }  // while
message_done:
  return ptr;
failure:
  ptr = nullptr;
  goto message_done;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* Results::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:routeguide.Results)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // string res = 1;
  if (!this->_internal_res().empty()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_res().data(), static_cast<int>(this->_internal_res().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "routeguide.Results.res");
    target = stream->WriteStringMaybeAliased(
        1, this->_internal_res(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:routeguide.Results)
  return target;
}

size_t Results::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:routeguide.Results)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // string res = 1;
  if (!this->_internal_res().empty()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_res());
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData Results::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSizeCheck,
    Results::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*Results::GetClassData() const { return &_class_data_; }

void Results::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to,
                      const ::PROTOBUF_NAMESPACE_ID::Message& from) {
  static_cast<Results *>(to)->MergeFrom(
      static_cast<const Results &>(from));
}


void Results::MergeFrom(const Results& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:routeguide.Results)
  GOOGLE_DCHECK_NE(&from, this);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (!from._internal_res().empty()) {
    _internal_set_res(from._internal_res());
  }
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void Results::CopyFrom(const Results& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:routeguide.Results)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Results::IsInitialized() const {
  return true;
}

void Results::InternalSwap(Results* other) {
  using std::swap;
  auto* lhs_arena = GetArenaForAllocation();
  auto* rhs_arena = other->GetArenaForAllocation();
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      &res_, lhs_arena,
      &other->res_, rhs_arena
  );
}

::PROTOBUF_NAMESPACE_ID::Metadata Results::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_route_5fguide_2eproto_getter, &descriptor_table_route_5fguide_2eproto_once,
      file_level_metadata_route_5fguide_2eproto[1]);
}

// @@protoc_insertion_point(namespace_scope)
}  // namespace routeguide
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::routeguide::Jsonfile* Arena::CreateMaybeMessage< ::routeguide::Jsonfile >(Arena* arena) {
  return Arena::CreateMessageInternal< ::routeguide::Jsonfile >(arena);
}
template<> PROTOBUF_NOINLINE ::routeguide::Results* Arena::CreateMaybeMessage< ::routeguide::Results >(Arena* arena) {
  return Arena::CreateMessageInternal< ::routeguide::Results >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
