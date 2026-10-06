// Part of OpenRAW (our own code, MIT). Used by tools/build_dng_validate.sh.
// Stub of dng_xmp_sdk for building dng_validate WITHOUT Adobe's XMP Toolkit.
// All XMP operations are no-ops: structural DNG/TIFF validation is unaffected;
// only XMP metadata handling is disabled (openraw writes no XMP anyway).
#include "dng_xmp_sdk.h"
#include "dng_string.h"
#include "dng_memory.h"
dng_xmp_sdk::dng_xmp_sdk () : fPrivate (NULL) {}
dng_xmp_sdk::dng_xmp_sdk (const dng_xmp_sdk &) : fPrivate (NULL) {}
dng_xmp_sdk::~dng_xmp_sdk () {}
void dng_xmp_sdk::InitializeSDK (dng_xmp_namespace *, const char *) {}
void dng_xmp_sdk::TerminateSDK () {}
bool dng_xmp_sdk::HasMeta () const { return false; }
void * dng_xmp_sdk::GetPrivateMeta () { return NULL; }
void dng_xmp_sdk::Parse (dng_host &, const char *, uint32) {}
bool dng_xmp_sdk::Exists (const char *, const char *) const { return false; }
void dng_xmp_sdk::AppendArrayItem (const char *, const char *, const char *, bool, bool) {}
int32 dng_xmp_sdk::CountArrayItems (const char *, const char *) const { return 0; }
bool dng_xmp_sdk::HasNameSpace (const char *) const { return false; }
void dng_xmp_sdk::Remove (const char *, const char *) {}
void dng_xmp_sdk::RemoveProperties (const char *) {}
bool dng_xmp_sdk::IsEmptyString (const char *, const char *) { return true; }
bool dng_xmp_sdk::IsEmptyArray (const char *, const char *) { return true; }
void dng_xmp_sdk::ComposeArrayItemPath (const char *, const char *, int32, dng_string &) const {}
void dng_xmp_sdk::ComposeStructFieldPath (const char *, const char *, const char *, const char *, dng_string &) const {}
bool dng_xmp_sdk::GetNamespacePrefix (const char *, dng_string &) const { return false; }
bool dng_xmp_sdk::GetString (const char *, const char *, dng_string &) const { return false; }
void dng_xmp_sdk::ValidateStringList (const char *, const char *) {}
bool dng_xmp_sdk::GetStringList (const char *, const char *, dng_string_list &) const { return false; }
bool dng_xmp_sdk::GetAltLangDefault (const char *, const char *, dng_string &) const { return false; }
bool dng_xmp_sdk::GetStructField (const char *, const char *, const char *, const char *, dng_string &) const { return false; }
void dng_xmp_sdk::Set (const char *, const char *, const char *) {}
void dng_xmp_sdk::SetString (const char *, const char *, const dng_string &) {}
void dng_xmp_sdk::SetStringList (const char *, const char *, const dng_string_list &, bool) {}
void dng_xmp_sdk::SetAltLangDefault (const char *, const char *, const dng_string &) {}
void dng_xmp_sdk::SetStructField (const char *, const char *, const char *, const char *, const char *) {}
void dng_xmp_sdk::DeleteStructField (const char *, const char *, const char *, const char *) {}
dng_memory_block * dng_xmp_sdk::Serialize (dng_memory_allocator &, bool, uint32, uint32, bool, bool) const { return NULL; }
void dng_xmp_sdk::PackageForJPEG (dng_memory_allocator &, AutoPtr<dng_memory_block> &, AutoPtr<dng_memory_block> &, dng_string &) const {}
void dng_xmp_sdk::MergeFromJPEG (const dng_xmp_sdk *) {}
void dng_xmp_sdk::ReplaceXMP (dng_xmp_sdk *) {}
bool dng_xmp_sdk::IteratePaths (IteratePathsCallback *, void *, const char *, const char *) { return false; }
void dng_xmp_sdk::ClearMeta () {}
void dng_xmp_sdk::MakeMeta () {}
void dng_xmp_sdk::NeedMeta () {}
const char *XMP_NS_TIFF = "http://ns.stub/TIFF/";
const char *XMP_NS_EXIF = "http://ns.stub/EXIF/";
const char *XMP_NS_PHOTOSHOP = "http://ns.stub/PHOTOSHOP/";
const char *XMP_NS_XAP = "http://ns.stub/XAP/";
const char *XMP_NS_XAP_RIGHTS = "http://ns.stub/XAP_RIGHTS/";
const char *XMP_NS_DC = "http://ns.stub/DC/";
const char *XMP_NS_XMP_NOTE = "http://ns.stub/XMP_NOTE/";
const char *XMP_NS_MM = "http://ns.stub/MM/";
const char *XMP_NS_CRS = "http://ns.stub/CRS/";
const char *XMP_NS_CRSS = "http://ns.stub/CRSS/";
const char *XMP_NS_LCP = "http://ns.stub/LCP/";
const char *XMP_NS_AUX = "http://ns.stub/AUX/";
const char *XMP_NS_IPTC = "http://ns.stub/IPTC/";
const char *XMP_NS_IPTC_EXT = "http://ns.stub/IPTC_EXT/";
const char *XMP_NS_CRX = "http://ns.stub/CRX/";
const char *XMP_NS_DNG = "http://ns.stub/DNG/";
