from rest_framework import viewsets
from rest_framework.response import Response

from demo.constants import *
from serving.api.predict import PredictionServer

prediction_server_obj = PredictionServer()


class PredictionViewSet(viewsets.ViewSet):

    def post(self, request):
        if len(set(request.data.keys()).intersection(set(MANDATORY_INPUT_FIELDS))) != len(MANDATORY_INPUT_FIELDS):
            return Response({'status': 'error', 'message': 'Missing mandatory fields'})
        return Response({'status': 'success', 'data': prediction_server_obj.predict(request.data)})
