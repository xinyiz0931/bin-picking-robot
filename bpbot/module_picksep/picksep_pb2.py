# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: picksep.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\rpicksep.proto\"\x1a\n\x07ImgPath\x12\x0f\n\x07imgpath\x18\x01 \x01(\t\".\n\tActionCls\x12\x11\n\tpickorsep\x18\x01 \x01(\x05\x12\x0e\n\x06\x61\x63tion\x18\x02 \x01(\x0c\"\x18\n\x06\x41\x63tion\x12\x0e\n\x06\x61\x63tion\x18\x01 \x01(\x0c\x32\xbb\x01\n\x07PickSep\x12\'\n\rinfer_picknet\x12\x08.ImgPath\x1a\n.ActionCls\"\x00\x12\x32\n\x18infer_picknet_sepnet_pos\x12\x08.ImgPath\x1a\n.ActionCls\"\x00\x12#\n\x0cinfer_sepnet\x12\x08.ImgPath\x1a\x07.Action\"\x00\x12.\n\x14infer_picknet_sepnet\x12\x08.ImgPath\x1a\n.ActionCls\"\x00\x62\x06proto3')



_IMGPATH = DESCRIPTOR.message_types_by_name['ImgPath']
_ACTIONCLS = DESCRIPTOR.message_types_by_name['ActionCls']
_ACTION = DESCRIPTOR.message_types_by_name['Action']
ImgPath = _reflection.GeneratedProtocolMessageType('ImgPath', (_message.Message,), {
  'DESCRIPTOR' : _IMGPATH,
  '__module__' : 'picksep_pb2'
  # @@protoc_insertion_point(class_scope:ImgPath)
  })
_sym_db.RegisterMessage(ImgPath)

ActionCls = _reflection.GeneratedProtocolMessageType('ActionCls', (_message.Message,), {
  'DESCRIPTOR' : _ACTIONCLS,
  '__module__' : 'picksep_pb2'
  # @@protoc_insertion_point(class_scope:ActionCls)
  })
_sym_db.RegisterMessage(ActionCls)

Action = _reflection.GeneratedProtocolMessageType('Action', (_message.Message,), {
  'DESCRIPTOR' : _ACTION,
  '__module__' : 'picksep_pb2'
  # @@protoc_insertion_point(class_scope:Action)
  })
_sym_db.RegisterMessage(Action)

_PICKSEP = DESCRIPTOR.services_by_name['PickSep']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _IMGPATH._serialized_start=17
  _IMGPATH._serialized_end=43
  _ACTIONCLS._serialized_start=45
  _ACTIONCLS._serialized_end=91
  _ACTION._serialized_start=93
  _ACTION._serialized_end=117
  _PICKSEP._serialized_start=120
  _PICKSEP._serialized_end=307
# @@protoc_insertion_point(module_scope)
