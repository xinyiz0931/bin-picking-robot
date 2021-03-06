# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: asp.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\tasp.proto\"\x17\n\x06\x44Value\x12\r\n\x05value\x18\x01 \x01(\x01\"\t\n\x07NoValue\"+\n\x08\x41SPInput\x12\x0f\n\x07imgpath\x18\x01 \x01(\t\x12\x0e\n\x06grasps\x18\x02 \x01(\x0c\"\x1a\n\tASPOutput\x12\r\n\x05probs\x18\x01 \x01(\x0c\"*\n\x06\x41GPair\x12\x0e\n\x06\x61\x63tion\x18\x01 \x01(\x05\x12\x10\n\x08graspidx\x18\x02 \x01(\x05\x32\x92\x01\n\x03\x41SP\x12\x34\n\x19\x61\x63tion_success_prediction\x12\t.ASPInput\x1a\n.ASPOutput\"\x00\x12/\n\x16\x61\x63tion_grasp_inference\x12\n.ASPOutput\x1a\x07.AGPair\"\x00\x12$\n\rset_threshold\x12\x07.DValue\x1a\x08.NoValue\"\x00\x62\x06proto3')



_DVALUE = DESCRIPTOR.message_types_by_name['DValue']
_NOVALUE = DESCRIPTOR.message_types_by_name['NoValue']
_ASPINPUT = DESCRIPTOR.message_types_by_name['ASPInput']
_ASPOUTPUT = DESCRIPTOR.message_types_by_name['ASPOutput']
_AGPAIR = DESCRIPTOR.message_types_by_name['AGPair']
DValue = _reflection.GeneratedProtocolMessageType('DValue', (_message.Message,), {
  'DESCRIPTOR' : _DVALUE,
  '__module__' : 'asp_pb2'
  # @@protoc_insertion_point(class_scope:DValue)
  })
_sym_db.RegisterMessage(DValue)

NoValue = _reflection.GeneratedProtocolMessageType('NoValue', (_message.Message,), {
  'DESCRIPTOR' : _NOVALUE,
  '__module__' : 'asp_pb2'
  # @@protoc_insertion_point(class_scope:NoValue)
  })
_sym_db.RegisterMessage(NoValue)

ASPInput = _reflection.GeneratedProtocolMessageType('ASPInput', (_message.Message,), {
  'DESCRIPTOR' : _ASPINPUT,
  '__module__' : 'asp_pb2'
  # @@protoc_insertion_point(class_scope:ASPInput)
  })
_sym_db.RegisterMessage(ASPInput)

ASPOutput = _reflection.GeneratedProtocolMessageType('ASPOutput', (_message.Message,), {
  'DESCRIPTOR' : _ASPOUTPUT,
  '__module__' : 'asp_pb2'
  # @@protoc_insertion_point(class_scope:ASPOutput)
  })
_sym_db.RegisterMessage(ASPOutput)

AGPair = _reflection.GeneratedProtocolMessageType('AGPair', (_message.Message,), {
  'DESCRIPTOR' : _AGPAIR,
  '__module__' : 'asp_pb2'
  # @@protoc_insertion_point(class_scope:AGPair)
  })
_sym_db.RegisterMessage(AGPair)

_ASP = DESCRIPTOR.services_by_name['ASP']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _DVALUE._serialized_start=13
  _DVALUE._serialized_end=36
  _NOVALUE._serialized_start=38
  _NOVALUE._serialized_end=47
  _ASPINPUT._serialized_start=49
  _ASPINPUT._serialized_end=92
  _ASPOUTPUT._serialized_start=94
  _ASPOUTPUT._serialized_end=120
  _AGPAIR._serialized_start=122
  _AGPAIR._serialized_end=164
  _ASP._serialized_start=167
  _ASP._serialized_end=313
# @@protoc_insertion_point(module_scope)
